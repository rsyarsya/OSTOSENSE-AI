from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Mapping


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class Arm(StringEnum):
    LEAK_GRADUAL = "LEAK_GRADUAL"
    SAFE = "SAFE"
    LEAK_SUDDEN = "LEAK_SUDDEN"
    FIELD = "FIELD"


class EndReason(StringEnum):
    CEILING_REACHED = "CEILING_REACHED"
    LEAK_CONFIRMED = "LEAK_CONFIRMED"
    MANUAL_STOP = "MANUAL_STOP"


class CapQuality(StringEnum):
    OK = "OK"
    DISCONNECTED = "DISCONNECTED"
    ADC_SATURATED = "ADC_SATURATED"
    BASELINE_INVALID = "BASELINE_INVALID"
    DATA_GAP = "DATA_GAP"
    WARMING_UP = "WARMING_UP"


class LigQuality(StringEnum):
    OK = "OK"
    DISCONNECTED = "DISCONNECTED"
    ADC_SATURATED = "ADC_SATURATED"
    BASELINE_INVALID = "BASELINE_INVALID"
    DATA_GAP = "DATA_GAP"
    WARMING_UP = "WARMING_UP"


class SystemQuality(StringEnum):
    INITIALIZING = "INITIALIZING"
    NORMAL = "NORMAL"
    ML_UNAVAILABLE = "ML_UNAVAILABLE"
    FAILSAFE_DEGRADED = "FAILSAFE_DEGRADED"
    UNSAFE = "UNSAFE"


class EventType(StringEnum):
    INJECTION_START = "INJECTION_START"
    INJECTION_END = "INJECTION_END"
    PHYSICAL_LEAK_OBSERVED = "PHYSICAL_LEAK_OBSERVED"
    LEAK_FLAG_FIRST = "LEAK_FLAG_FIRST"
    LEAK_FLAG_CONFIRMED = "LEAK_FLAG_CONFIRMED"
    LIG_CALIBRATION_STARTED = "LIG_CALIBRATION_STARTED"
    LIG_CALIBRATION_PASSED = "LIG_CALIBRATION_PASSED"
    LIG_CALIBRATION_FAILED = "LIG_CALIBRATION_FAILED"
    BAG_EMPTIED_LOGGED = "BAG_EMPTIED_LOGGED"
    ALERT_RAISED = "ALERT_RAISED"
    ALERT_ACKNOWLEDGED = "ALERT_ACKNOWLEDGED"
    ALERT_IGNORED = "ALERT_IGNORED"
    MANUAL_SESSION_RESET = "MANUAL_SESSION_RESET"
    DEVICE_RESTART = "DEVICE_RESTART"


def _validate_id(name: str, value: str, *, optional: bool = False) -> None:
    if optional and value == "":
        return
    if not value or any(character in value for character in (",", "\n", "\r")):
        raise ValueError(f"{name} must be a non-empty, single-line CSV-safe string")


def _validate_timestamp(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative Unix epoch millisecond integer")


def _validate_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_enum(name: str, value: object, enum_type: type[Enum]) -> None:
    if not isinstance(value, enum_type):
        raise ValueError(f"{name} must be a {enum_type.__name__} value")


def aggregate_system_quality(
    cap_quality: CapQuality, lig_quality: LigQuality
) -> SystemQuality:
    if cap_quality is CapQuality.WARMING_UP or lig_quality is LigQuality.WARMING_UP:
        return SystemQuality.INITIALIZING
    cap_valid = cap_quality is CapQuality.OK
    lig_valid = lig_quality is LigQuality.OK
    if cap_valid and lig_valid:
        return SystemQuality.NORMAL
    if not cap_valid and lig_valid:
        return SystemQuality.ML_UNAVAILABLE
    if cap_valid and not lig_valid:
        return SystemQuality.FAILSAFE_DEGRADED
    return SystemQuality.UNSAFE


@dataclass(frozen=True)
class SessionRecord:
    FIELDS: ClassVar[tuple[str, ...]] = (
        "session_id",
        "arm",
        "bag_id",
        "sensor_id",
        "device_id",
        "fluid_type",
        "operator_id",
        "baseline_value",
        "baseline_std",
        "start_timestamp",
        "end_timestamp",
        "end_reason",
        "model_version",
        "firmware_version",
    )

    session_id: str
    arm: Arm
    bag_id: str
    sensor_id: str
    device_id: str
    fluid_type: str
    operator_id: str
    baseline_value: float
    baseline_std: float
    start_timestamp: int
    end_timestamp: int
    end_reason: EndReason
    model_version: str
    firmware_version: str

    def __post_init__(self) -> None:
        _validate_enum("arm", self.arm, Arm)
        _validate_enum("end_reason", self.end_reason, EndReason)
        for name in (
            "session_id",
            "bag_id",
            "sensor_id",
            "device_id",
            "fluid_type",
            "operator_id",
            "firmware_version",
        ):
            _validate_id(name, getattr(self, name))
        _validate_id("model_version", self.model_version, optional=True)
        _validate_finite("baseline_value", self.baseline_value)
        _validate_finite("baseline_std", self.baseline_std)
        if self.baseline_std < 0:
            raise ValueError("baseline_std must be non-negative")
        _validate_timestamp("start_timestamp", self.start_timestamp)
        _validate_timestamp("end_timestamp", self.end_timestamp)
        if self.end_timestamp < self.start_timestamp:
            raise ValueError("end_timestamp must be greater than or equal to start_timestamp")

    def to_row(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "arm": self.arm.value,
            "bag_id": self.bag_id,
            "sensor_id": self.sensor_id,
            "device_id": self.device_id,
            "fluid_type": self.fluid_type,
            "operator_id": self.operator_id,
            "baseline_value": self.baseline_value,
            "baseline_std": self.baseline_std,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "end_reason": self.end_reason.value,
            "model_version": self.model_version,
            "firmware_version": self.firmware_version,
        }


@dataclass(frozen=True)
class SampleRecord:
    FIELDS: ClassVar[tuple[str, ...]] = (
        "timestamp",
        "session_id",
        "capacitance_raw",
        "lig_raw",
        "cap_quality",
        "lig_quality",
        "system_quality",
        "activity_state",
        "orientation_position",
    )

    timestamp: int
    session_id: str
    capacitance_raw: float
    lig_raw: float
    cap_quality: CapQuality
    lig_quality: LigQuality
    system_quality: SystemQuality
    activity_state: str = ""
    orientation_position: str = ""

    @classmethod
    def create(
        cls,
        *,
        timestamp: int,
        session_id: str,
        capacitance_raw: float,
        lig_raw: float,
        cap_quality: CapQuality,
        lig_quality: LigQuality,
        activity_state: str = "",
        orientation_position: str = "",
    ) -> "SampleRecord":
        return cls(
            timestamp=timestamp,
            session_id=session_id,
            capacitance_raw=capacitance_raw,
            lig_raw=lig_raw,
            cap_quality=cap_quality,
            lig_quality=lig_quality,
            system_quality=aggregate_system_quality(cap_quality, lig_quality),
            activity_state=activity_state,
            orientation_position=orientation_position,
        )

    def __post_init__(self) -> None:
        _validate_enum("cap_quality", self.cap_quality, CapQuality)
        _validate_enum("lig_quality", self.lig_quality, LigQuality)
        _validate_enum("system_quality", self.system_quality, SystemQuality)
        _validate_timestamp("timestamp", self.timestamp)
        _validate_id("session_id", self.session_id)
        _validate_finite("capacitance_raw", self.capacitance_raw)
        _validate_finite("lig_raw", self.lig_raw)
        _validate_id("activity_state", self.activity_state, optional=True)
        _validate_id("orientation_position", self.orientation_position, optional=True)
        expected = aggregate_system_quality(self.cap_quality, self.lig_quality)
        if self.system_quality is not expected:
            raise ValueError(
                f"system_quality must be derived from channel quality: expected {expected.value}"
            )

    def to_row(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "capacitance_raw": self.capacitance_raw,
            "lig_raw": self.lig_raw,
            "cap_quality": self.cap_quality.value,
            "lig_quality": self.lig_quality.value,
            "system_quality": self.system_quality.value,
            "activity_state": self.activity_state,
            "orientation_position": self.orientation_position,
        }


@dataclass(frozen=True)
class EventRecord:
    FIELDS: ClassVar[tuple[str, ...]] = (
        "event_id",
        "session_id",
        "timestamp",
        "event_type",
        "event_metadata",
    )

    event_id: str
    session_id: str
    timestamp: int
    event_type: EventType
    event_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        _validate_enum("event_type", self.event_type, EventType)
        _validate_id("event_id", self.event_id)
        _validate_id("session_id", self.session_id)
        _validate_timestamp("timestamp", self.timestamp)
        if not isinstance(self.event_metadata, Mapping):
            raise ValueError("event_metadata must be a JSON object")
        try:
            json.dumps(self.event_metadata, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("event_metadata must contain JSON-safe finite values") from error

    def to_row(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "event_metadata": json.dumps(
                self.event_metadata,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        }
