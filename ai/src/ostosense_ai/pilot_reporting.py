"""Auditable progress-report figures for OSTOSENSE pilot and synthetic evidence.

The report deliberately keeps two evidence sources separate:

* real, unlabeled pilot logger data: feature-processing flow and descriptive
  sensor correlation only;
* ``SYNTHETIC_PIPELINE_TEST_ONLY`` labeled data: optimizer trace and validation
  Macro F1 mechanics only.

No figure generated here is a production-performance or clinical claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import tempfile
from pathlib import Path
from typing import Any

from ostosense_ai import evaluation, pilot_data, training

REPORTER_VERSION = "0.1.0"
PNG_DPI = 300
TRACE_COLUMNS: tuple[str, ...] = (
    "iteration",
    "objective",
    "objective_normalized",
    "validation_accuracy",
    "validation_macro_f1",
)
REPORT_FILES = (
    "optimization_trace.csv",
    "optimization_trace_manifest.json",
    "01_alur_ekstraksi_fitur.png",
    "02_matriks_korelasi_sensor_real.png",
    "03_jejak_optimisasi_olr_sintetis.png",
    "04_panel_ai_ostosense.png",
)
_REQUIRED_MODULES = ("numpy", "scipy", "sklearn", "mord", "matplotlib")
_DEPENDENCY_HINT = (
    "install the optional reporting stack with `pip install -e .[pipeline]` "
    "(numpy, scipy, scikit-learn, mord, matplotlib)"
)

SYNTHETIC_WARNING = (
    "SYNTHETIC_PIPELINE_TEST_ONLY: jejak ini menguji mekanik optimisasi dan "
    "validasi pipeline, bukan performa OSTOSENSE pada penggunaan nyata."
)
REAL_WARNING = (
    "Data pilot nyata tanpa label; korelasi bersifat deskriptif dan tidak "
    "menunjukkan sebab-akibat atau akurasi model."
)


class PilotReportingError(ValueError):
    """Reporting validation failure; destination artifacts stay untouched."""


def _load_stack() -> tuple[Any, ...]:
    missing = [name for name in _REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            f"optional dependencies are required for pilot reporting "
            f"(missing: {', '.join(missing)}); {_DEPENDENCY_HINT}"
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import patches
    from mord import LogisticAT
    from mord.threshold_based import (
        grad_margin,
        obj_margin,
        threshold_predict,
        threshold_proba,
    )
    from scipy import optimize
    from sklearn.preprocessing import StandardScaler

    return (
        np,
        plt,
        patches,
        optimize,
        StandardScaler,
        LogisticAT,
        obj_margin,
        grad_margin,
        threshold_predict,
        threshold_proba,
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PilotReportingError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PilotReportingError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise PilotReportingError(f"{label} must be a JSON object")
    return value


def _read_correlation(path: Path) -> list[list[float]]:
    if not path.is_file():
        raise PilotReportingError(f"missing correlation matrix: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(pilot_data.CORRELATION_COLUMNS):
            raise PilotReportingError("correlation matrix header is not canonical")
        rows = list(reader)
    if [row["sensor"] for row in rows] != list(pilot_data.SENSOR_CHANNELS):
        raise PilotReportingError("correlation matrix row order is not canonical")
    matrix: list[list[float]] = []
    for row in rows:
        values: list[float] = []
        for channel in pilot_data.SENSOR_CHANNELS:
            try:
                value = float(row[channel])
            except (TypeError, ValueError) as error:
                raise PilotReportingError("correlation matrix contains malformed data") from error
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise PilotReportingError("correlation matrix values must be finite in [-1, 1]")
            values.append(value)
        matrix.append(values)
    if len(matrix) != len(pilot_data.SENSOR_CHANNELS):
        raise PilotReportingError("correlation matrix must be 5x5")
    for row_index, row in enumerate(matrix):
        if abs(row[row_index] - 1.0) > 1e-10:
            raise PilotReportingError("correlation matrix diagonal must equal 1")
        for column_index in range(len(row)):
            if abs(row[column_index] - matrix[column_index][row_index]) > 1e-10:
                raise PilotReportingError("correlation matrix must be symmetric")
    return matrix


def _validate_pilot_artifacts(pilot_dir: Path) -> tuple[dict[str, Any], list[list[float]]]:
    manifest = _load_json(pilot_dir / "pilot_manifest.json", "pilot_manifest.json")
    if manifest.get("pilot_preparer_version") != pilot_data.PILOT_PREPARER_VERSION:
        raise PilotReportingError("pilot_preparer_version is not supported")
    if manifest.get("dataset_origin") != pilot_data.DATASET_ORIGIN:
        raise PilotReportingError("pilot dataset must be REAL_PILOT_UNLABELED")
    output_hashes = manifest.get("output_sha256")
    if not isinstance(output_hashes, dict):
        raise PilotReportingError("pilot output_sha256 must be an object")
    for file_name, expected in output_hashes.items():
        path = pilot_dir / file_name
        if not isinstance(expected, str) or not path.is_file() or _sha256_file(path) != expected:
            raise PilotReportingError(f"pilot artifact hash mismatch: {file_name}")
    correlation = manifest.get("correlation")
    if not isinstance(correlation, dict):
        raise PilotReportingError("pilot correlation metadata is missing")
    included = correlation.get("included_sessions")
    excluded = correlation.get("excluded_sessions")
    if not isinstance(included, list) or len(included) != 10:
        raise PilotReportingError("the real correlation figure requires 10 operational sessions")
    if excluded != ["P006"]:
        raise PilotReportingError("P006 must be the only correlation-excluded session")
    matrix = _read_correlation(pilot_dir / "sensor_correlation_median.csv")
    return manifest, matrix


def _trace_row(
    np,
    parameters,
    iteration: int,
    initial_objective: float,
    scaled_validation,
    validation_targets,
    objective_function,
    objective_args,
    threshold_predict,
    lower_triangle,
) -> dict[str, float | int]:
    objective = float(objective_function(parameters, *objective_args))
    feature_count = scaled_validation.shape[1]
    beta = parameters[:feature_count]
    theta = lower_triangle.dot(parameters[feature_count:])
    predictions = np.asarray(
        threshold_predict(scaled_validation, beta, theta), dtype=np.int64
    )
    metrics = evaluation.evaluate_predictions(validation_targets.tolist(), predictions.tolist())
    accuracy = float(np.mean(predictions == validation_targets))
    return {
        "iteration": iteration,
        "objective": objective,
        "objective_normalized": objective / initial_objective,
        "validation_accuracy": accuracy,
        "validation_macro_f1": float(metrics["macro_f1"]),
    }


def build_optimizer_trace(
    matrix_dir: str | Path,
    training_config_path: str | Path,
) -> tuple[list[dict[str, float | int]], dict[str, Any]]:
    """Re-run the pinned L-BFGS-B fit with callbacks and verify mord parity."""

    (
        np,
        _plt,
        _patches,
        optimize,
        StandardScaler,
        LogisticAT,
        obj_margin,
        grad_margin,
        threshold_predict,
        threshold_proba,
    ) = _load_stack()

    matrix_dir = Path(matrix_dir)
    config = training.load_config(training_config_path)
    training.validate_training_config(config)
    csv_path = matrix_dir / "model_matrix.csv"
    manifest_path = matrix_dir / "matrix_manifest.json"
    matrix_manifest = training._load_json_object(manifest_path, "matrix_manifest.json")
    rows = training._read_matrix_rows(csv_path)
    training._validate_matrix(matrix_manifest, rows, csv_path, config)

    development = [row for row in rows if row["dataset_partition"] == "development"]
    validation = [row for row in rows if row["dataset_partition"] == "validation"]
    if not development or not validation:
        raise PilotReportingError("optimizer trace requires development and validation rows")
    if {int(row["risk_label_index"]) for row in development} != {0, 1, 2, 3}:
        raise PilotReportingError("development rows must contain all four classes")
    if {int(row["risk_label_index"]) for row in validation} != {0, 1, 2, 3}:
        raise PilotReportingError("validation rows must contain all four classes")

    development_features = np.asarray(
        [[float(row[column]) for column in training.FEATURE_COLUMNS] for row in development],
        dtype=np.float64,
    )
    development_targets = np.asarray(
        [int(row["risk_label_index"]) for row in development], dtype=np.int64
    )
    validation_features = np.asarray(
        [[float(row[column]) for column in training.FEATURE_COLUMNS] for row in validation],
        dtype=np.float64,
    )
    validation_targets = np.asarray(
        [int(row["risk_label_index"]) for row in validation], dtype=np.int64
    )

    scaler = StandardScaler().fit(development_features)
    scaled_development = scaler.transform(development_features)
    scaled_validation = scaler.transform(validation_features)
    n_classes = 4
    feature_count = scaled_development.shape[1]
    lower_triangle = np.zeros((n_classes - 1, n_classes - 1))
    lower_triangle[np.tril_indices(n_classes - 1)] = 1.0
    weights = np.ones((n_classes, n_classes - 1))
    initial = np.zeros(feature_count + n_classes - 1)
    initial[feature_count:] = np.arange(n_classes - 1)
    bounds = [(None, None)] * (feature_count + 1) + [(0, None)] * (n_classes - 2)
    objective_args = (
        scaled_development,
        development_targets,
        float(config["alpha"]),
        n_classes,
        weights,
        lower_triangle,
        None,
    )
    initial_objective = float(obj_margin(initial, *objective_args))
    if not math.isfinite(initial_objective) or initial_objective <= 0.0:
        raise PilotReportingError("initial optimizer objective must be positive and finite")

    trace: list[dict[str, float | int]] = []
    parameter_history: list[Any] = []

    def record(parameters) -> None:
        parameter_copy = np.asarray(parameters, dtype=np.float64).copy()
        trace.append(
            _trace_row(
                np,
                parameter_copy,
                len(trace),
                initial_objective,
                scaled_validation,
                validation_targets,
                obj_margin,
                objective_args,
                threshold_predict,
                lower_triangle,
            )
        )
        parameter_history.append(parameter_copy)

    record(initial)
    solution = optimize.minimize(
        obj_margin,
        initial,
        method="L-BFGS-B",
        jac=grad_margin,
        bounds=bounds,
        options={"maxiter": int(config["max_iter"]), "disp": False},
        args=objective_args,
        tol=1e-12,
        callback=record,
    )
    if not parameter_history or not np.array_equal(parameter_history[-1], solution.x):
        record(solution.x)

    objectives = [float(row["objective"]) for row in trace]
    if any(
        later > earlier + 1e-8 * max(1.0, abs(earlier))
        for earlier, later in zip(objectives, objectives[1:])
    ):
        raise PilotReportingError("optimizer callback objective is not non-increasing")

    final_beta = np.asarray(solution.x[:feature_count], dtype=np.float64)
    final_theta = lower_triangle.dot(solution.x[feature_count:])
    canonical = LogisticAT(
        alpha=float(config["alpha"]), max_iter=int(config["max_iter"])
    ).fit(scaled_development, development_targets)
    canonical_beta = np.asarray(canonical.coef_, dtype=np.float64).ravel()
    canonical_theta = np.asarray(canonical.theta_, dtype=np.float64).ravel()
    combined = np.vstack((scaled_development, scaled_validation))
    traced_probabilities = np.asarray(
        threshold_proba(combined, final_beta, final_theta), dtype=np.float64
    )
    canonical_probabilities = np.asarray(canonical.predict_proba(combined), dtype=np.float64)
    traced_classes = np.asarray(
        threshold_predict(combined, final_beta, final_theta), dtype=np.int64
    )
    canonical_classes = np.asarray(canonical.predict(combined), dtype=np.int64)

    beta_max_difference = float(np.max(np.abs(final_beta - canonical_beta)))
    theta_max_difference = float(np.max(np.abs(final_theta - canonical_theta)))
    probability_max_difference = float(
        np.max(np.abs(traced_probabilities - canonical_probabilities))
    )
    class_parity = bool(np.array_equal(traced_classes, canonical_classes))
    tolerance = 1e-6
    if max(beta_max_difference, theta_max_difference, probability_max_difference) > tolerance:
        raise PilotReportingError("instrumented optimizer does not reproduce canonical mord")
    if not class_parity:
        raise PilotReportingError("instrumented optimizer class predictions differ from mord")

    parity = {
        "solver_method": "L-BFGS-B",
        "solver_success": bool(solution.success),
        "solver_status": int(solution.status),
        "solver_message": str(solution.message),
        "optimizer_iteration_count": len(trace) - 1,
        "development_row_count": len(development),
        "validation_row_count": len(validation),
        "beta_max_absolute_difference": beta_max_difference,
        "theta_max_absolute_difference": theta_max_difference,
        "probability_max_absolute_difference": probability_max_difference,
        "class_prediction_parity": class_parity,
        "parity_tolerance": tolerance,
    }
    return trace, parity


def _write_trace(path: Path, trace: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_COLUMNS)
        writer.writeheader()
        for row in trace:
            writer.writerow(
                {
                    key: str(int(value))
                    if key == "iteration"
                    else format(float(value), ".12g")
                    for key, value in row.items()
                }
            )


def _style_axis(ax) -> None:
    ax.set_facecolor("#FFFFFF")
    for spine in ax.spines.values():
        spine.set_color("#D6DCE5")
    ax.tick_params(colors="#263238", labelsize=9)


def _draw_flow(ax, patches, *, compact: bool = False) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    blue = "#1769AA"
    pale = "#EAF3FA"
    gold = "#F2B134"
    charcoal = "#263238"
    stages = [
        "Data sensor mentah\n5 kanal, 10 Hz",
        "Pemeriksaan mutu\nskema, waktu, status",
        "Ringkasan median\n10 sampel menjadi 1 Hz",
        "Perbandingan baseline\n20 detik awal (sementara)",
        "Jendela analisis\n120 detik, langkah 10 detik",
        "15 ciri kapasitif\n3 kanal x 5 ciri",
        "Model OLR\nSafe - Monitor - Caution - Urgent",
    ]
    left = 0.035
    right = 0.965
    gap = 0.012
    width = (right - left - gap * (len(stages) - 1)) / len(stages)
    y = 0.53 if compact else 0.56
    height = 0.25 if compact else 0.28
    for index, label in enumerate(stages):
        x = left + index * (width + gap)
        face = "#FFF4D6" if index == len(stages) - 1 else pale
        edge = gold if index == len(stages) - 1 else blue
        box = patches.FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            color=charcoal,
            fontsize=7.1 if compact else 8.4,
            fontweight="bold" if index in (0, len(stages) - 1) else "normal",
        )
        if index < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + width + gap * 0.86, y + height / 2),
                xytext=(x + width + gap * 0.14, y + height / 2),
                arrowprops={"arrowstyle": "->", "color": blue, "lw": 1.3},
            )

    branch_x = left
    branch_y = 0.13 if compact else 0.14
    branch_w = width * 2.35
    branch_h = 0.18
    box = patches.FancyBboxPatch(
        (branch_x, branch_y),
        branch_w,
        branch_h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor="#F4F5F7",
        edgecolor="#59636E",
        linewidth=1.3,
    )
    ax.add_patch(box)
    ax.text(
        branch_x + branch_w / 2,
        branch_y + branch_h / 2,
        "Jalur pengaman terpisah\nSensor resistif dalam + luar -> peringatan kontak cairan",
        ha="center",
        va="center",
        fontsize=7.3 if compact else 8.4,
        color=charcoal,
    )
    ax.annotate(
        "",
        xy=(branch_x + branch_w * 0.25, branch_y + branch_h),
        xytext=(left + width * 0.5, y),
        arrowprops={"arrowstyle": "->", "color": "#59636E", "lw": 1.2},
    )
    ax.text(
        0.99,
        0.035,
        "Data nyata saat ini belum memiliki label risiko per jendela.",
        ha="right",
        va="bottom",
        fontsize=7.2 if compact else 8.2,
        color="#59636E",
    )


def _draw_heatmap(ax, np, matrix: list[list[float]], *, compact: bool = False) -> None:
    labels = ["Resistif\ndalam", "Resistif\nluar", "Kapasitif\n4", "Kapasitif\n5", "Kapasitif\n7"]
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(5), labels, fontsize=7.1 if compact else 8.5)
    ax.set_yticks(range(5), labels, fontsize=7.1 if compact else 8.5)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    for row in range(5):
        for column in range(5):
            value = matrix[row][column]
            color = "white" if abs(value) >= 0.62 else "#263238"
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=7.3 if compact else 9.0,
                fontweight="bold" if row == column else "normal",
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def _draw_trace(ax, trace: list[dict[str, float | int]], *, compact: bool = False) -> None:
    blue = "#1769AA"
    gold = "#E19720"
    iterations = [int(row["iteration"]) for row in trace]
    objectives = [float(row["objective_normalized"]) for row in trace]
    macro_f1 = [float(row["validation_macro_f1"]) for row in trace]
    ax.plot(iterations, objectives, color=blue, lw=2.0, marker="o", ms=2.8, label="Objektif training (normalisasi)")
    ax.set_xlabel("Iterasi optimisasi L-BFGS-B", fontsize=8 if compact else 9.5)
    ax.set_ylabel("Objektif / nilai awal", color=blue, fontsize=8 if compact else 9.5)
    ax.tick_params(axis="y", labelcolor=blue)
    ax.grid(axis="both", color="#E6E9ED", linewidth=0.8)
    _style_axis(ax)
    other = ax.twinx()
    other.plot(iterations, macro_f1, color=gold, lw=2.0, marker="s", ms=2.8, label="Macro F1 validation sintetis")
    other.set_ylabel("Macro F1 validation sintetis", color=gold, fontsize=8 if compact else 9.5)
    other.tick_params(axis="y", labelcolor=gold, labelsize=8 if compact else 9)
    other.set_ylim(0, 1.03)
    other.spines["top"].set_visible(False)
    other.spines["right"].set_color("#D6DCE5")
    handles = ax.get_lines() + other.get_lines()
    ax.legend(handles, [line.get_label() for line in handles], loc="best", frameon=False, fontsize=7 if compact else 8.5)


def _save_figure(fig, path: Path) -> None:
    fig.savefig(
        path,
        dpi=PNG_DPI,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "OSTOSENSE pilot_reporting", "Title": path.stem},
    )


def _make_figures(
    stage: Path,
    matrix: list[list[float]],
    trace: list[dict[str, float | int]],
    included_session_count: int,
) -> None:
    np, plt, patches, *_rest = _load_stack()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(13.3, 5.6), constrained_layout=True)
    fig.suptitle("Alur Pengolahan Data AI OSTOSENSE", fontsize=15, fontweight="bold", color="#263238")
    _draw_flow(ax, patches)
    _save_figure(fig, stage / "01_alur_ekstraksi_fitur.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 6.8), constrained_layout=True)
    fig.suptitle("Matriks Korelasi Sensor pada Data Pilot Nyata", fontsize=14, fontweight="bold", color="#263238")
    image = _draw_heatmap(ax, np, matrix)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Korelasi Spearman", fontsize=9)
    ax.set_title(
        f"Median per sesi; {included_session_count} sesi operasional memiliki bobot sama\n"
        "P006 dikeluarkan karena merupakan uji gangguan sensor\n"
        "Data nyata tanpa label; korelasi deskriptif, bukan akurasi model atau sebab-akibat",
        fontsize=9.5,
        pad=14,
        color="#59636E",
    )
    _save_figure(fig, stage / "02_matriks_korelasi_sensor_real.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.8), constrained_layout=True)
    fig.suptitle("Jejak Optimisasi Model Ordinal Logistic Regression", fontsize=14, fontweight="bold", color="#263238")
    _draw_trace(ax, trace)
    ax.set_title(
        "Development untuk fitting; validation hanya untuk pemantauan mekanik\n"
        "SYNTHETIC_PIPELINE_TEST_ONLY - bukan performa OSTOSENSE pada penggunaan nyata",
        fontsize=9.5,
        color="#59636E",
        pad=10,
    )
    _save_figure(fig, stage / "03_jejak_optimisasi_olr_sintetis.png")
    plt.close(fig)

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.84, 1.16], width_ratios=[0.95, 1.05])
    flow_ax = fig.add_subplot(grid[0, :])
    heat_ax = fig.add_subplot(grid[1, 0])
    trace_ax = fig.add_subplot(grid[1, 1])
    fig.suptitle("Ringkasan Pipeline AI OSTOSENSE", fontsize=17, fontweight="bold", color="#263238")
    _draw_flow(flow_ax, patches, compact=True)
    image = _draw_heatmap(heat_ax, np, matrix, compact=True)
    heat_ax.set_title(
        f"Data nyata tanpa label: median korelasi {included_session_count} sesi",
        fontsize=10,
        pad=9,
    )
    colorbar = fig.colorbar(image, ax=heat_ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=7)
    _draw_trace(trace_ax, trace, compact=True)
    trace_ax.set_title(
        "Data sintetis: jejak optimisasi OLR\nSYNTHETIC_PIPELINE_TEST_ONLY",
        fontsize=10,
        pad=9,
    )
    _save_figure(fig, stage / "04_panel_ai_ostosense.png")
    plt.close(fig)


def _check_outputs(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {output_dir}")
    existing = [name for name in REPORT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "output already contains reporting artifacts: "
            + ", ".join(existing)
            + "; pass overwrite=True or --overwrite to replace them"
        )


def generate_progress_report(
    pilot_dir: str | Path,
    synthetic_matrix_dir: str | Path,
    training_config_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate trace evidence and four 300-dpi PNG progress figures."""

    pilot_dir = Path(pilot_dir)
    synthetic_matrix_dir = Path(synthetic_matrix_dir)
    training_config_path = Path(training_config_path)
    output_dir = Path(output_dir)
    _check_outputs(output_dir, overwrite)
    pilot_manifest, matrix = _validate_pilot_artifacts(pilot_dir)
    trace, parity = build_optimizer_trace(synthetic_matrix_dir, training_config_path)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ostosense-report-", dir=output_dir.parent) as name:
        stage = Path(name)
        _write_trace(stage / "optimization_trace.csv", trace)
        included_count = len(pilot_manifest["correlation"]["included_sessions"])
        _make_figures(stage, matrix, trace, included_count)
        output_hashes = {
            file_name: _sha256_file(stage / file_name)
            for file_name in REPORT_FILES
            if file_name != "optimization_trace_manifest.json"
        }
        matrix_manifest_path = synthetic_matrix_dir / "matrix_manifest.json"
        matrix_csv_path = synthetic_matrix_dir / "model_matrix.csv"
        manifest = {
            "reporter_version": REPORTER_VERSION,
            "evidence_separation": {
                "real_unlabeled": "feature flow and descriptive sensor correlation",
                "synthetic_pipeline_test_only": "optimizer trace and validation Macro F1 mechanics",
            },
            "input_sha256": {
                "pilot_manifest_json": _sha256_file(pilot_dir / "pilot_manifest.json"),
                "sensor_correlation_median_csv": _sha256_file(
                    pilot_dir / "sensor_correlation_median.csv"
                ),
                "synthetic_model_matrix_csv": _sha256_file(matrix_csv_path),
                "synthetic_matrix_manifest_json": _sha256_file(matrix_manifest_path),
                "training_config_json": _sha256_file(training_config_path),
            },
            "output_sha256": output_hashes,
            "png_dpi": PNG_DPI,
            "trace_columns": list(TRACE_COLUMNS),
            "optimizer_parity": parity,
            "real_correlation_session_count": included_count,
            "real_correlation_excluded_sessions": pilot_manifest["correlation"][
                "excluded_sessions"
            ],
            "warnings": [REAL_WARNING, SYNTHETIC_WARNING],
        }
        (stage / "optimization_trace_manifest.json").write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for file_name in REPORT_FILES:
            (stage / file_name).replace(output_dir / file_name)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ostosense_ai.pilot_reporting",
        description="Generate auditable real-pilot and synthetic-mechanics progress figures.",
    )
    parser.add_argument("--pilot", required=True, help="Prepared real-pilot artifact directory.")
    parser.add_argument("--synthetic-matrix", required=True, help="Synthetic model-matrix directory.")
    parser.add_argument("--training-config", required=True, help="Pinned training config JSON.")
    parser.add_argument("--output", required=True, help="Output directory for trace and PNG files.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing report artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = generate_progress_report(
        args.pilot,
        args.synthetic_matrix,
        args.training_config,
        args.output,
        overwrite=args.overwrite,
    )
    parity = manifest["optimizer_parity"]
    print(
        "pilot-reporting: wrote 4 PNGs at 300 dpi; "
        f"optimizer iterations={parity['optimizer_iteration_count']}, "
        f"class parity={parity['class_prediction_parity']}"
    )
    print(REAL_WARNING)
    print(SYNTHETIC_WARNING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
