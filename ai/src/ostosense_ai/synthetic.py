"""Deterministic synthetic raw-data generator for OSTOSENSE pipeline testing.

SYNTHETIC_PIPELINE_TEST_ONLY.

This module writes contract-valid ``sessions.csv``, ``samples.csv``, and
``events.csv`` through the existing ``ostosense_contract`` logger, plus a
deterministic ``manifest.json`` sidecar. Its only purpose is to exercise
mechanical pipeline plumbing: CSV generation, Data Contract v1.1 compliance,
deterministic (byte-identical) output, scenario coverage, and quality-state
handling.

The output is *not* evidence of sensor behavior, clinical validity, model
performance, or OSTOSENSE performance, and must never support any performance
claim. Every tunable numeric parameter lives in the JSON config and carries an
explicit provenance classification (see ``ALLOWED_PROVENANCE``). Fixed unit and
formatting constants (milliseconds per second, CSV rounding decimals) are
definitional, not experimental parameters.

Capacitive and LIG responsibilities stay separate: the capacitive series is the
future AI-risk input, while LIG only carries synthetic contact/fail-safe
information and is never turned into an AI feature here. No derived features,
four-class risk labels, or B1/B2/B3 boundaries are produced.

CLI::

    python -m ostosense_ai.synthetic \\
        --config ai/configs/synthetic-v0.2.json \\
        --output /tmp/ostosense-synthetic-smoke \\
        --seed 20260722
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ostosense_contract import (
    Arm,
    CapQuality,
    EndReason,
    EventRecord,
    EventType,
    LigQuality,
    SampleRecord,
    SessionRecord,
    Tier1CsvLogger,
)

GENERATOR_VERSION = "0.2.0"
DATA_CONTRACT_VERSION = "v1.1"
DATASET_ORIGIN = "SYNTHETIC_PIPELINE_TEST_ONLY"
SYNTHETIC_WARNING = (
    "Synthetic data generated for mechanical pipeline testing only. It does "
    "not represent sensor behavior, clinical validity, or model/OSTOSENSE "
    "performance, and must never be used to support any performance claim."
)

ALLOWED_PROVENANCE = frozenset(
    {
        "CONTRACT_DERIVED",
        "ENGINEERING_TEST_ONLY",
        "LITERATURE_VERIFIED",
        "PILOT_PENDING",
    }
)
PROVENANCE_LEGEND = {
    "CONTRACT_DERIVED": "Fixed by OSTOSENSE Data Contract v1.1.",
    "ENGINEERING_TEST_ONLY": "Arbitrary synthetic value chosen only to drive "
    "the pipeline harness; carries no physical meaning.",
    "LITERATURE_VERIFIED": "Backed by an exact source already present in the "
    "repository (none are used in this config).",
    "PILOT_PENDING": "Placeholder awaiting a real Tier 1 pilot measurement.",
}

MS_PER_SECOND = 1000
_VALUE_DECIMALS = 6
_OUTPUT_FILES = ("sessions.csv", "samples.csv", "events.csv", "manifest.json")

_KIND_TO_ARM = {
    "safe": Arm.SAFE,
    "gradual": Arm.LEAK_GRADUAL,
    "sudden": Arm.LEAK_SUDDEN,
    "fault_cap_disconnect": Arm.SAFE,
    "fault_data_gap": Arm.SAFE,
    "fault_calibration": Arm.SAFE,
    "fault_both_disconnect": Arm.SAFE,
}
_LEAK_KINDS = frozenset({"gradual", "sudden"})
_QC_KINDS = frozenset(kind for kind in _KIND_TO_ARM if kind.startswith("fault_"))
_SCENARIO_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


# --------------------------------------------------------------------------- #
# Config loading and provenance validation
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict[str, Any]:
    """Load a generator config JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_numeric_paths(value: Any, prefix: str = "") -> Iterator[str]:
    """Yield dotted paths of every numeric leaf; list indices collapse to ``[]``."""
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield prefix
        return
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_numeric_paths(child, child_prefix)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_numeric_paths(item, f"{prefix}[]")


def validate_config_provenance(config: dict[str, Any]) -> None:
    """Require every numeric config parameter to carry a valid provenance label.

    Raises ``ValueError`` if a numeric parameter is unclassified, if a label is
    outside ``ALLOWED_PROVENANCE``, or if the provenance map lists a key that is
    not an actual numeric parameter (keeps the map honest and in sync).
    """
    numeric_paths = set(
        _iter_numeric_paths(
            {key: value for key, value in config.items() if key != "provenance"}
        )
    )
    provenance = config.get("provenance", {})

    missing = sorted(numeric_paths - provenance.keys())
    if missing:
        raise ValueError(f"numeric parameters missing provenance: {missing}")

    extra = sorted(provenance.keys() - numeric_paths)
    if extra:
        raise ValueError(f"provenance lists non-existent numeric parameters: {extra}")

    invalid = sorted(
        key for key, label in provenance.items() if label not in ALLOWED_PROVENANCE
    )
    if invalid:
        raise ValueError(f"invalid provenance labels on: {invalid}")


def _require_int(name: str, value: Any, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _require_number(
    name: str, value: Any, *, minimum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _require_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_fraction(name: str, value: Any) -> float:
    result = _require_number(name, value, minimum=0.0)
    if result >= 1.0:
        raise ValueError(f"{name} must be < 1.0")
    return result


def _require_exact_keys(
    name: str, value: dict[str, Any], expected: set[str]
) -> None:
    missing = sorted(expected - value.keys())
    if missing:
        raise ValueError(f"{name} is missing required keys: {missing}")
    extra = sorted(value.keys() - expected)
    if extra:
        raise ValueError(f"{name} contains unsupported keys: {extra}")


def validate_config(config: dict[str, Any]) -> None:
    """Fail fast on unsupported or internally inconsistent generator config."""
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")

    required_keys = {
        "config_id",
        "data_contract_version",
        "base_timestamp_ms",
        "session_gap_s",
        "sampling_rate_hz",
        "warmup_s",
        "baseline_window_s",
        "min_full_window_s",
        "identity",
        "signal",
        "phases",
        "events",
        "scenarios",
        "provenance",
    }
    _require_exact_keys("config", config, required_keys)

    for key in ("identity", "signal", "phases", "events", "provenance"):
        if not isinstance(config[key], dict):
            raise ValueError(f"{key} must be a JSON object")

    _require_string("config_id", config["config_id"])
    _require_string("data_contract_version", config["data_contract_version"])
    if config["data_contract_version"] != DATA_CONTRACT_VERSION:
        raise ValueError(f"data_contract_version must be {DATA_CONTRACT_VERSION}")

    _require_int("base_timestamp_ms", config["base_timestamp_ms"], minimum=0)
    _require_int("session_gap_s", config["session_gap_s"], minimum=1)
    sampling_rate = _require_int(
        "sampling_rate_hz", config["sampling_rate_hz"], minimum=1
    )
    if sampling_rate != 1:
        raise ValueError("sampling_rate_hz must be exactly 1 for generator v0.2")
    _require_int("warmup_s", config["warmup_s"], minimum=1)
    _require_int("baseline_window_s", config["baseline_window_s"], minimum=1)
    _require_int("min_full_window_s", config["min_full_window_s"], minimum=1)

    identity = config["identity"]
    identity_keys = {"device_id", "operator_id", "firmware_version", "fluid_type"}
    _require_exact_keys("identity", identity, identity_keys)
    for key in identity_keys:
        _require_string(f"identity.{key}", identity.get(key))

    signal = config["signal"]
    signal_level_keys = {
        "cap_baseline",
        "cap_gradual_rise",
        "cap_sudden_jump",
        "lig_dry_level",
        "lig_contact_level",
    }
    signal_noise_keys = {"cap_noise_sigma", "lig_noise_sigma"}
    _require_exact_keys("signal", signal, signal_level_keys | signal_noise_keys)
    for key in signal_level_keys:
        _require_number(f"signal.{key}", signal.get(key))
    for key in signal_noise_keys:
        _require_number(f"signal.{key}", signal.get(key), minimum=0.0)

    phases = config["phases"]
    phase_keys = {
        "gradual_dry_fraction",
        "gradual_contact_fraction",
        "sudden_contact_fraction",
        "leak_flag_delay_s",
        "leak_confirm_delay_s",
        "fault_start_fraction",
        "fault_duration_s",
    }
    _require_exact_keys("phases", phases, phase_keys)
    gradual_dry = _require_fraction(
        "phases.gradual_dry_fraction", phases.get("gradual_dry_fraction")
    )
    gradual_contact = _require_fraction(
        "phases.gradual_contact_fraction", phases.get("gradual_contact_fraction")
    )
    if gradual_dry >= gradual_contact:
        raise ValueError(
            "gradual_dry_fraction must be less than gradual_contact_fraction"
        )
    _require_fraction(
        "phases.sudden_contact_fraction", phases.get("sudden_contact_fraction")
    )
    flag_delay = _require_int(
        "phases.leak_flag_delay_s", phases.get("leak_flag_delay_s"), minimum=0
    )
    confirm_delay = _require_int(
        "phases.leak_confirm_delay_s", phases.get("leak_confirm_delay_s"), minimum=0
    )
    if confirm_delay < flag_delay:
        raise ValueError("leak_confirm_delay_s must be >= leak_flag_delay_s")
    _require_fraction("phases.fault_start_fraction", phases.get("fault_start_fraction"))
    _require_int("phases.fault_duration_s", phases.get("fault_duration_s"), minimum=1)

    events = config["events"]
    event_keys = {
        "injection_volume_ml_per_step",
        "target_flow_ml_min",
        "injection_step_s",
    }
    _require_exact_keys("events", events, event_keys)
    for key in ("injection_volume_ml_per_step", "target_flow_ml_min"):
        value = _require_number(f"events.{key}", events.get(key), minimum=0.0)
        if value == 0.0:
            raise ValueError(f"events.{key} must be > 0")
    _require_int("events.injection_step_s", events.get("injection_step_s"), minimum=1)

    if not isinstance(config["scenarios"], list) or not config["scenarios"]:
        raise ValueError("scenarios must be a non-empty list")

    validate_config_provenance(config)
    _validate_scenarios(config)


def _validate_scenarios(config: dict[str, Any]) -> None:
    phases = config["phases"]
    warmup = config["warmup_s"]
    baseline_window = config["baseline_window_s"]
    min_full_window = config["min_full_window_s"]
    seen_ids: set[str] = set()
    max_duration = 0

    for index, scenario in enumerate(config["scenarios"]):
        prefix = f"scenarios[{index}]"
        if not isinstance(scenario, dict):
            raise ValueError(f"{prefix} must be a JSON object")

        scenario_id = _require_string(
            f"{prefix}.scenario_id", scenario.get("scenario_id")
        )
        if not _SCENARIO_ID_PATTERN.fullmatch(scenario_id):
            raise ValueError(f"{prefix}.scenario_id must be a lowercase slug")
        if scenario_id in seen_ids:
            raise ValueError(f"duplicate scenario_id: {scenario_id}")
        seen_ids.add(scenario_id)

        kind = _require_string(f"{prefix}.kind", scenario.get("kind"))
        if kind not in _KIND_TO_ARM:
            raise ValueError(f"{prefix}.kind is unsupported: {kind}")
        scenario_keys = {
            "scenario_id",
            "arm",
            "kind",
            "count",
            "duration_s",
            "bag_id",
            "sensor_id",
        }
        if kind in _LEAK_KINDS:
            scenario_keys.add("physical_observation_delay_s")
        _require_exact_keys(prefix, scenario, scenario_keys)
        try:
            arm = Arm(scenario.get("arm"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{prefix}.arm is not a Data Contract arm") from error
        if arm is not _KIND_TO_ARM[kind]:
            raise ValueError(f"{prefix}.arm does not match kind {kind}")

        _require_int(f"{prefix}.count", scenario.get("count"), minimum=1)
        duration = _require_int(
            f"{prefix}.duration_s", scenario.get("duration_s"), minimum=1
        )
        _require_string(f"{prefix}.bag_id", scenario.get("bag_id"))
        _require_string(f"{prefix}.sensor_id", scenario.get("sensor_id"))

        is_leak = kind in _LEAK_KINDS
        if is_leak:
            observation_delay = _require_int(
                f"{prefix}.physical_observation_delay_s",
                scenario.get("physical_observation_delay_s"),
                minimum=0,
            )
        else:
            if "physical_observation_delay_s" in scenario:
                raise ValueError(f"{prefix} must not define physical observation delay")
            observation_delay = 0

        times = _phase_times(duration, config)
        if kind == "gradual":
            dry_end = times["t_inj"]
            contact_time: int | None = times["t_contact_gradual"]
        elif kind == "sudden":
            dry_end = times["t_contact_sudden"]
            contact_time = times["t_contact_sudden"]
        elif kind == "safe":
            dry_end = duration
            contact_time = None
        else:
            dry_end = times["fault_start"]
            contact_time = None

        if kind not in _QC_KINDS:
            required_dry = warmup + max(baseline_window, min_full_window)
            if dry_end < required_dry:
                raise ValueError(f"{prefix} has too few valid dry samples")
        else:
            required_baseline = warmup + baseline_window
            if dry_end < required_baseline:
                raise ValueError(f"{prefix} fault begins before baseline completes")
            if times["fault_end"] >= duration:
                raise ValueError(f"{prefix} fault interval reaches session end")

        if contact_time is not None:
            latest_event = contact_time + max(
                observation_delay, phases["leak_confirm_delay_s"]
            )
            if latest_event >= duration:
                raise ValueError(f"{prefix} leak event occurs after session end")
            if kind == "gradual" and contact_time - times["t_inj"] <= 1:
                raise ValueError(f"{prefix} leaves no valid injection interval")

        max_duration = max(max_duration, duration)

    if config["session_gap_s"] <= max_duration:
        raise ValueError("session_gap_s must exceed every scenario duration")


def _canonical_config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Deterministic session planning and signal synthesis
# --------------------------------------------------------------------------- #
def _round_value(raw: float) -> float:
    value = round(float(raw), _VALUE_DECIMALS)
    if value == 0.0:  # normalize -0.0 for byte-stable CSV output
        return 0.0
    return value


def _session_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    interval_ms = MS_PER_SECOND
    gap_ms = config["session_gap_s"] * MS_PER_SECOND
    base = config["base_timestamp_ms"]
    plan: list[dict[str, Any]] = []
    index = 0
    for scenario in config["scenarios"]:
        kind = scenario["kind"]
        for occurrence in range(1, scenario["count"] + 1):
            plan.append(
                {
                    "index": index,
                    "session_id": f"syn-{scenario['scenario_id']}-{occurrence:03d}",
                    "arm": scenario["arm"],
                    "scenario_id": scenario["scenario_id"],
                    "kind": kind,
                    "duration_s": scenario["duration_s"],
                    "bag_id": scenario["bag_id"],
                    "sensor_id": scenario["sensor_id"],
                    "physical_observation_delay_s": scenario.get(
                        "physical_observation_delay_s", 0
                    ),
                    "start_timestamp": base + index * gap_ms,
                    "interval_ms": interval_ms,
                }
            )
            index += 1
    return plan


def _phase_times(duration: int, config: dict[str, Any]) -> dict[str, int]:
    phases = config["phases"]
    fault_start = round(duration * phases["fault_start_fraction"])
    return {
        "t_inj": round(duration * phases["gradual_dry_fraction"]),
        "t_contact_gradual": round(duration * phases["gradual_contact_fraction"]),
        "t_contact_sudden": round(duration * phases["sudden_contact_fraction"]),
        "fault_start": fault_start,
        "fault_end": fault_start + phases["fault_duration_s"],
    }


def _generate_session(
    spec: dict[str, Any], config: dict[str, Any], rng: np.random.Generator
) -> tuple[SessionRecord, list[SampleRecord], list[EventRecord]]:
    duration = spec["duration_s"]
    warmup = config["warmup_s"]
    interval = spec["interval_ms"]
    start = spec["start_timestamp"]
    kind = spec["kind"]
    signal = config["signal"]
    times = _phase_times(duration, config)

    cap_noise = rng.normal(0.0, signal["cap_noise_sigma"], size=duration)
    lig_noise = rng.normal(0.0, signal["lig_noise_sigma"], size=duration)

    is_leak = kind in _LEAK_KINDS
    t_contact = (
        times["t_contact_gradual"]
        if kind == "gradual"
        else times["t_contact_sudden"]
    )

    samples: list[SampleRecord] = []
    baseline_caps: list[float] = []
    for i in range(duration):
        cap_quality = CapQuality.OK
        lig_quality = LigQuality.OK

        if kind == "gradual":
            if i < times["t_inj"]:
                cap = signal["cap_baseline"]
            else:
                span = max(1, times["t_contact_gradual"] - times["t_inj"])
                ramp = min(1.0, (i - times["t_inj"]) / span)
                cap = signal["cap_baseline"] + signal["cap_gradual_rise"] * ramp
        elif kind == "sudden":
            jump = (
                signal["cap_sudden_jump"] if i >= times["t_contact_sudden"] else 0.0
            )
            cap = signal["cap_baseline"] + jump
        else:
            cap = signal["cap_baseline"]
        cap += cap_noise[i]

        if is_leak and i >= t_contact:
            lig = signal["lig_contact_level"] + lig_noise[i]
        else:
            lig = signal["lig_dry_level"] + lig_noise[i]

        if i < warmup:
            cap_quality = CapQuality.WARMING_UP
            lig_quality = LigQuality.WARMING_UP
        elif kind == "fault_cap_disconnect":
            if times["fault_start"] <= i < times["fault_end"]:
                cap_quality = CapQuality.DISCONNECTED
        elif kind == "fault_data_gap":
            if times["fault_start"] <= i < times["fault_end"]:
                continue  # intentional contract-valid gap: omit rows
            if i == times["fault_end"]:
                cap_quality = CapQuality.DATA_GAP
        elif kind == "fault_both_disconnect":
            if times["fault_start"] <= i < times["fault_end"]:
                cap_quality = CapQuality.DISCONNECTED
                lig_quality = LigQuality.DISCONNECTED
        elif kind == "fault_calibration":
            lig_quality = LigQuality.BASELINE_INVALID

        cap_value = _round_value(cap)
        samples.append(
            SampleRecord.create(
                timestamp=start + i * interval,
                session_id=spec["session_id"],
                capacitance_raw=cap_value,
                lig_raw=_round_value(lig),
                cap_quality=cap_quality,
                lig_quality=lig_quality,
            )
        )
        if (
            cap_quality is CapQuality.OK
            and len(baseline_caps) < config["baseline_window_s"]
        ):
            baseline_caps.append(cap_value)

    baseline_value = _round_value(float(np.median(baseline_caps)))
    baseline_std = _round_value(float(np.std(baseline_caps)))

    if is_leak:
        end_reason = EndReason.LEAK_CONFIRMED
    elif kind == "safe":
        end_reason = EndReason.CEILING_REACHED
    else:
        end_reason = EndReason.MANUAL_STOP

    session = SessionRecord(
        session_id=spec["session_id"],
        arm=Arm(spec["arm"]),
        bag_id=spec["bag_id"],
        sensor_id=spec["sensor_id"],
        device_id=config["identity"]["device_id"],
        fluid_type=config["identity"]["fluid_type"],
        operator_id=config["identity"]["operator_id"],
        baseline_value=baseline_value,
        baseline_std=baseline_std,
        start_timestamp=start,
        end_timestamp=samples[-1].timestamp,
        end_reason=end_reason,
        model_version="",
        firmware_version=config["identity"]["firmware_version"],
    )
    events = _build_events(spec, config, times)
    return session, samples, events


def _leak_events(
    start: int,
    t_contact: int,
    physical_observation_delay_s: int,
    phases: dict[str, Any],
    operator_id: str,
) -> list[tuple[int, EventType, dict[str, Any]]]:
    consecutive_samples = (
        phases["leak_confirm_delay_s"] - phases["leak_flag_delay_s"] + 1
    )
    return [
        (
            start
            + (t_contact + physical_observation_delay_s) * MS_PER_SECOND,
            EventType.PHYSICAL_LEAK_OBSERVED,
            {"observation_method": "SYNTHETIC", "operator_id": operator_id},
        ),
        (
            start
            + (t_contact + phases["leak_flag_delay_s"]) * MS_PER_SECOND,
            EventType.LEAK_FLAG_FIRST,
            {"source": "synthetic_lig"},
        ),
        (
            start
            + (t_contact + phases["leak_confirm_delay_s"]) * MS_PER_SECOND,
            EventType.LEAK_FLAG_CONFIRMED,
            {"consecutive_samples": consecutive_samples},
        ),
    ]


def _build_events(
    spec: dict[str, Any], config: dict[str, Any], times: dict[str, int]
) -> list[EventRecord]:
    kind = spec["kind"]
    start = spec["start_timestamp"]
    warmup = config["warmup_s"]
    phases = config["phases"]
    events_cfg = config["events"]
    operator_id = config["identity"]["operator_id"]

    raw: list[tuple[int, EventType, dict[str, Any]]] = [
        (
            start,
            EventType.LIG_CALIBRATION_STARTED,
            {"attempt": 1, "reason": "NEW_SESSION"},
        )
    ]
    calibration_ts = start + warmup * MS_PER_SECOND
    if kind == "fault_calibration":
        raw.append(
            (
                calibration_ts,
                EventType.LIG_CALIBRATION_FAILED,
                {
                    "duration_ms": warmup * MS_PER_SECOND,
                    "sample_count": warmup,
                    "failure_reason": "SYNTHETIC_FAULT_INJECTION",
                    "validity_rule_version": "synthetic-v0.2",
                },
            )
        )
    else:
        raw.append(
            (
                calibration_ts,
                EventType.LIG_CALIBRATION_PASSED,
                {
                    "duration_ms": warmup * MS_PER_SECOND,
                    "sample_count": warmup,
                    "validity_rule_version": "synthetic-v0.2",
                },
            )
        )

    if kind == "gradual":
        t_inj = times["t_inj"]
        step = min(
            events_cfg["injection_step_s"],
            times["t_contact_gradual"] - t_inj - 1,
        )
        volume = events_cfg["injection_volume_ml_per_step"]
        flow = events_cfg["target_flow_ml_min"]
        raw.append(
            (
                start + t_inj * MS_PER_SECOND,
                EventType.INJECTION_START,
                {"delivery_mode": "synthetic_stepwise", "target_flow_ml_min": flow},
            )
        )
        raw.append(
            (
                start + (t_inj + step) * MS_PER_SECOND,
                EventType.INJECTION_END,
                {
                    "delivered_volume_ml": volume,
                    "cumulative_volume_ml": volume,
                    "measured_flow_ml_min": flow,
                },
            )
        )
        raw.extend(
            _leak_events(
                start,
                times["t_contact_gradual"],
                spec["physical_observation_delay_s"],
                phases,
                operator_id,
            )
        )
    elif kind == "sudden":
        raw.extend(
            _leak_events(
                start,
                times["t_contact_sudden"],
                spec["physical_observation_delay_s"],
                phases,
                operator_id,
            )
        )
    raw.sort(key=lambda item: item[0])
    return [
        EventRecord(
            event_id=f"{spec['session_id']}-evt-{order:03d}",
            session_id=spec["session_id"],
            timestamp=timestamp,
            event_type=event_type,
            event_metadata=metadata,
        )
        for order, (timestamp, event_type, metadata) in enumerate(raw)
    ]


# --------------------------------------------------------------------------- #
# Top-level generation
# --------------------------------------------------------------------------- #
def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        output_dir / name
        for name in _OUTPUT_FILES
        if (output_dir / name).exists()
    ]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"output already contains generated artifacts: {names}; "
            "pass overwrite=True or --overwrite to replace them"
        )
    for target in existing:
        target.unlink()


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    path.write_text(text, encoding="utf-8")


def generate(
    config: dict[str, Any],
    seed: int,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate the synthetic dataset and manifest; return the manifest dict."""
    validate_config(config)
    _require_int("seed", seed, minimum=0)

    plan = _session_plan(config)
    session_ids = [spec["session_id"] for spec in plan]
    if len(set(session_ids)) != len(session_ids):
        raise ValueError("session plan produced duplicate session_id values")

    children = np.random.SeedSequence(seed).spawn(len(plan))
    generated_sessions = []
    for spec, child in zip(plan, children):
        session, samples, events = _generate_session(
            spec, config, np.random.default_rng(child)
        )
        if any(
            not session.start_timestamp <= event.timestamp <= session.end_timestamp
            for event in events
        ):
            raise RuntimeError(f"event outside session bounds: {spec['session_id']}")
        generated_sessions.append((spec, session, samples, events))
    output_dir = Path(output_dir)
    _prepare_output_dir(output_dir, overwrite=overwrite)
    logger = Tier1CsvLogger(output_dir)

    session_summaries: list[dict[str, Any]] = []
    scenario_counts: dict[str, int] = {}
    total_samples = 0
    total_events = 0
    for spec, session, samples, events in generated_sessions:
        for sample in samples:
            logger.append_sample(sample)
        for event in events:
            logger.append_event(event)
        logger.append_session(session)

        has_leak = any(
            event.event_type is EventType.PHYSICAL_LEAK_OBSERVED for event in events
        )
        session_summaries.append(
            {
                "session_id": spec["session_id"],
                "arm": spec["arm"],
                "scenario_id": spec["scenario_id"],
                "kind": spec["kind"],
                "bag_id": spec["bag_id"],
                "sensor_id": spec["sensor_id"],
                "physical_observation_delay_s": spec["physical_observation_delay_s"],
                "start_timestamp": session.start_timestamp,
                "end_timestamp": session.end_timestamp,
                "sample_count": len(samples),
                "event_count": len(events),
                "has_physical_leak": has_leak,
            }
        )
        key = f"{spec['arm']}/{spec['kind']}"
        scenario_counts[key] = scenario_counts.get(key, 0) + 1
        total_samples += len(samples)
        total_events += len(events)

    manifest = {
        "dataset_origin": DATASET_ORIGIN,
        "generator_version": GENERATOR_VERSION,
        "data_contract_version": config["data_contract_version"],
        "config_id": config["config_id"],
        "config_sha256": _canonical_config_hash(config),
        "seed": int(seed),
        "nominal_sampling_rate_hz": config["sampling_rate_hz"],
        "warning": SYNTHETIC_WARNING,
        "session_count": len(plan),
        "sample_count": total_samples,
        "event_count": total_events,
        "scenario_session_counts": scenario_counts,
        "sessions": session_summaries,
        "numeric_parameter_provenance": config["provenance"],
        "provenance_legend": PROVENANCE_LEGEND,
    }
    _write_manifest(output_dir / "manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ostosense_ai.synthetic",
        description="Deterministic SYNTHETIC_PIPELINE_TEST_ONLY raw-data generator.",
    )
    parser.add_argument(
        "--config", required=True, help="Path to a generator config JSON."
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for CSVs + manifest."
    )
    parser.add_argument(
        "--seed", required=True, type=int, help="Deterministic integer seed."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing generator artifacts in the output directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = generate(
        load_config(args.config), args.seed, args.output, overwrite=args.overwrite
    )
    print(
        f"{DATASET_ORIGIN}: wrote {manifest['session_count']} sessions, "
        f"{manifest['sample_count']} samples, {manifest['event_count']} events "
        f"to {args.output}"
    )
    print(manifest["warning"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
