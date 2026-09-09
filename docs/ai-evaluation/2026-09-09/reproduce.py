#!/usr/bin/env python3
"""Local, offline evaluation companion; never modifies repository or raw inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import warnings
from importlib.metadata import version

REPO = Path(__file__).resolve().parents[3]
RAW = None
MODEL_HASH = "ceb2922841431a38c274e0ebd5c350873a28a95f40a6f6aab2194a1603b93cef"
MODEL_NAMES = ["OLR", "Logistic Regression", "Decision Tree"]
CLASSES = ["Safe", "Monitor", "Caution", "Urgent"]
CHANNELS = ["Res_15", "Res_16", "Kap_4", "Kap_5", "Kap_7"]
COLORS = ["#2563A6", "#C47818", "#76843D"]
SYNTHETIC = "SYNTHETIC_PIPELINE_TEST_ONLY"
REAL = "REAL_PILOT_UNLABELED"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repo_input_hashes(repo):
    paths = list((repo / "ai/src/ostosense_ai").glob("*.py"))
    paths += list((repo / "ai/configs").glob("*.json"))
    paths += list((repo / "ai").glob("requirements-*.lock"))
    paths += [
        repo / name
        for name in (
            "ai/pyproject.toml",
            "scripts/build_engineering_demo.sh",
            "scripts/verify.sh",
            "ai/data-manifests/real-pilot-v0.1.inventory.json",
        )
    ]
    return {str(p.relative_to(repo)): sha(p) for p in sorted(paths)}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(cmd, repo, output, name):
    env = {
        **os.environ,
        "PYTHONPATH": str(repo / "ai/src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(output / "work/mplconfig"),
    }
    with (output / "logs" / name).open("w") as handle:
        result = subprocess.run(
            [str(x) for x in cmd],
            cwd=repo,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=300,
        )
    if result.returncode:
        raise RuntimeError(f"{name}: exit {result.returncode}; inspect the saved log")


def main(args):
    repo, out = args.repo.resolve(), args.output.resolve()
    if args.refresh_report:
        render_report(out, read_json(out / "analysis.json"))
        check_links(out)
        update_artifact_manifest(out)
        return
    if args.raw is None:
        raise ValueError("Pass --raw with the private input directory; raw data are not published.")
    raw = args.raw.resolve()
    if (out / "analysis.json").exists():
        raise FileExistsError(
            "Analysis already exists; choose a fresh --output directory"
        )
    for child in ("tables", "figures", "logs", "work", "checks"):
        (out / child).mkdir(parents=True, exist_ok=True)
    if Path(__file__).resolve() != out / "reproduce.py":
        shutil.copyfile(__file__, out / "reproduce.py")
    sys.path.insert(0, str(repo / "ai/src"))
    os.environ["MPLCONFIGDIR"] = str(out / "work/mplconfig")
    os.environ["MPLBACKEND"] = "Agg"
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandardScaler
    from ostosense_ai import evaluation, inference, pilot_reporting

    inv_path = repo / "ai/data-manifests/real-pilot-v0.1.inventory.json"
    input_hashes_before = repo_input_hashes(repo)
    inventory = read_json(inv_path)
    source_rows = []
    for entry in inventory["sessions"]:
        source = raw / entry["file_name"]
        digest = sha(source)
        if digest != entry["sha256"]:
            raise ValueError(f"Raw source mismatch: {entry['file_name']}")
        source_rows.append(
            {
                "session_id": entry["session_id"],
                "file_name": entry["file_name"],
                "sha256": digest,
                "expected_sha256": entry["sha256"],
                "match": True,
            }
        )
    write_csv(out / "tables/source_inventory.csv", source_rows)
    assert len(source_rows) == 11
    model_source = (
        repo / "ai/src/ostosense_ai/model_assets/engineering_v03/ordinal_model.json"
    )
    if sha(model_source) != MODEL_HASH:
        raise ValueError(
            "Bundled model differs from the model approved for this evaluation"
        )

    synthetic = out / "work/synthetic"
    python = repo / "ai/.venv/bin/python"
    if not (synthetic / "model/training_manifest.json").exists():
        run(
            [repo / "scripts/build_engineering_demo.sh", synthetic],
            repo,
            out,
            "synthetic.log",
        )
    if sha(synthetic / "model/ordinal_model.json") != MODEL_HASH:
        raise ValueError("Rebuilt OLR model does not match bundled SHA-256")
    run(
        [
            python,
            "-m",
            "ostosense_ai.model_evaluation",
            "--matrix",
            synthetic / "matrix",
            "--model",
            synthetic / "model",
            "--output",
            out / "work/canonical-validation",
            "--overwrite",
        ],
        repo,
        out,
        "canonical-validation.log",
    )
    run(
        [
            python,
            "-m",
            "ostosense_ai.pilot_data",
            "--input",
            raw,
            "--config",
            repo / "ai/configs/real-pilot-v0.1.json",
            "--output",
            out / "work/pilot",
            "--overwrite",
        ],
        repo,
        out,
        "real-pilot.log",
    )
    matrix = read_csv(synthetic / "matrix/model_matrix.csv")
    assert {r["dataset_partition"] for r in matrix} == {"development", "validation"}
    split = {
        part: [r for r in matrix if r["dataset_partition"] == part]
        for part in ("development", "validation")
    }
    grouping = {}
    for key in ("session_id", "bag_id", "sensor_id"):
        groups = {part: sorted({r[key] for r in rows}) for part, rows in split.items()}
        assert not set(groups["development"]) & set(groups["validation"])
        grouping[key] = groups
    assert [len(split[p]) for p in ("development", "validation")] == [76, 38]
    features = list(inference.FEATURE_COLUMNS)
    X = {
        p: np.array([[float(r[c]) for c in features] for r in rows])
        for p, rows in split.items()
    }
    y = {
        p: np.array([int(r["risk_label_index"]) for r in rows])
        for p, rows in split.items()
    }
    assert all(np.isfinite(v).all() for v in X.values())
    assert all(set(a.tolist()) == {0, 1, 2, 3} for a in y.values())
    model = read_json(synthetic / "model/ordinal_model.json")
    scaler = StandardScaler().fit(X["development"])
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=10000, class_weight=None)
    dt = DecisionTreeClassifier(
        criterion="gini",
        max_depth=3,
        min_samples_leaf=2,
        class_weight=None,
        random_state=20260909,
    )
    warning_rows = []
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        lr.fit(scaler.transform(X["development"]), y["development"])
        dt.fit(X["development"], y["development"])
        trace, parity = pilot_reporting.build_optimizer_trace(
            synthetic / "matrix", repo / "ai/configs/training-v0.1.json"
        )
    for item in captured:
        warning_rows.append(
            {"category": item.category.__name__, "message": str(item.message)}
        )
    assert parity["class_prediction_parity"]
    if not parity["solver_success"]:
        warning_rows.append(
            {"category": "OptimizerFailure", "message": parity["solver_message"]}
        )
    write_json(out / "checks/fit_warnings.json", warning_rows)

    specs = {
        "OLR": {
            "model_family": "mord.LogisticAT",
            "alpha": 1.0,
            "max_iter": 10000,
            "preprocessing": "StandardScaler fitted on Development only",
            "model_sha256": MODEL_HASH,
        },
        "Logistic Regression": {
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 10000,
            "class_weight": None,
            "preprocessing": "StandardScaler fitted on Development only",
        },
        "Decision Tree": {
            "criterion": "gini",
            "max_depth": 3,
            "min_samples_leaf": 2,
            "class_weight": None,
            "random_state": 20260909,
            "preprocessing": "none",
        },
    }
    write_json(out / "model_specifications.json", specs)
    write_json(
        out / "work/comparator_parameters.json",
        {
            "logistic_regression": {
                "classes": lr.classes_.tolist(),
                "coef": lr.coef_.tolist(),
                "intercept": lr.intercept_.tolist(),
                "n_iter": lr.n_iter_.tolist(),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
            },
            "decision_tree": {
                "classes": dt.classes_.tolist(),
                "children_left": dt.tree_.children_left.tolist(),
                "children_right": dt.tree_.children_right.tolist(),
                "feature": dt.tree_.feature.tolist(),
                "threshold": dt.tree_.threshold.tolist(),
                "value": dt.tree_.value.tolist(),
            },
        },
    )
    pred_rows, metric_rows, per_class_rows, metrics = [], [], [], {}
    cm_rows = []
    for name in MODEL_NAMES:
        metrics[name] = {}
        for part in ("development", "validation"):
            if name == "OLR":
                pred = np.array(
                    inference.predict_exported_model(model, X[part])[
                        "predicted_indices"
                    ]
                )
            elif name == "Logistic Regression":
                pred = lr.predict(scaler.transform(X[part]))
            else:
                pred = dt.predict(X[part])
            scores = evaluation.evaluate_predictions(y[part].tolist(), pred.tolist())
            accuracy = float(np.mean(y[part] == pred))
            cm = np.array(scores["confusion_matrix"])
            assert int(cm.sum()) == len(pred)
            assert abs(accuracy - np.trace(cm) / cm.sum()) < 1e-12
            # Independently verify per-class precision, recall and F1 from integer counts.
            manual_f1 = []
            for c, item in enumerate(scores["per_class"]):
                assert item["support"] == int((y[part] == c).sum()) == int(cm[c].sum())
                pr = float(cm[c, c] / cm[:, c].sum()) if cm[:, c].sum() else 0.0
                rc = float(cm[c, c] / cm[c].sum()) if cm[c].sum() else 0.0
                f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
                assert (
                    max(
                        abs(pr - item["precision"]),
                        abs(rc - item["recall"]),
                        abs(f1 - item["f1"]),
                    )
                    < 1e-12
                )
                manual_f1.append(f1)
                per_class_rows.append(
                    {
                        "model": name,
                        "partition": part,
                        **item,
                        "dataset_origin": SYNTHETIC,
                    }
                )
            assert abs(float(np.mean(manual_f1)) - scores["macro_f1"]) < 1e-12
            scores["accuracy"] = accuracy
            metrics[name][part] = scores
            metric_rows.append(
                {
                    "model": name,
                    "partition": part,
                    "n_windows": len(pred),
                    "n_sessions": len(grouping["session_id"][part]),
                    "accuracy": accuracy,
                    "macro_f1": scores["macro_f1"],
                    "quadratic_weighted_kappa": scores["quadratic_weighted_kappa"],
                    "dataset_origin": SYNTHETIC,
                }
            )
            for i, row in enumerate(split[part]):
                pred_rows.append(
                    {
                        "model": name,
                        "partition": part,
                        "window_id": row["window_id"],
                        "session_id": row["session_id"],
                        "bag_id": row["bag_id"],
                        "sensor_id": row["sensor_id"],
                        "true_index": int(y[part][i]),
                        "true_class": CLASSES[y[part][i]],
                        "predicted_index": int(pred[i]),
                        "predicted_class": CLASSES[pred[i]],
                        "dataset_origin": SYNTHETIC,
                    }
                )
            for i, label in enumerate(CLASSES):
                cm_rows.append(
                    {
                        "model": name,
                        "partition": part,
                        "true_class": label,
                        **{f"pred_{c}": int(cm[i, j]) for j, c in enumerate(CLASSES)},
                    }
                )
    canonical = read_json(out / "work/canonical-validation/validation_evaluation.json")[
        "pipeline_mechanics_metrics"
    ]
    assert (
        canonical["confusion_matrix"]
        == metrics["OLR"]["validation"]["confusion_matrix"]
    )
    # The canonical JSON writer rounds floats to 12 decimal places.
    for key in ("macro_f1", "quadratic_weighted_kappa"):
        assert abs(canonical[key] - metrics["OLR"]["validation"][key]) < 1e-12
    canonical_predictions = read_csv(
        out / "work/canonical-validation/validation_predictions.csv"
    )
    actual_predictions = [
        r for r in pred_rows if r["model"] == "OLR" and r["partition"] == "validation"
    ]
    assert [
        (r["window_id"], int(r["predicted_index"])) for r in canonical_predictions
    ] == [(r["window_id"], r["predicted_index"]) for r in actual_predictions]
    # Repeat independent comparator fits with frozen parameters and compare all predictions.
    lr2 = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=10000, class_weight=None
    ).fit(scaler.transform(X["development"]), y["development"])
    dt2 = DecisionTreeClassifier(
        criterion="gini",
        max_depth=3,
        min_samples_leaf=2,
        class_weight=None,
        random_state=20260909,
    ).fit(X["development"], y["development"])
    for part in split:
        assert np.array_equal(
            lr.predict(scaler.transform(X[part])),
            lr2.predict(scaler.transform(X[part])),
        )
        assert np.array_equal(dt.predict(X[part]), dt2.predict(X[part]))
    write_csv(out / "tables/model_metrics.csv", metric_rows)
    write_csv(out / "tables/per_class_metrics.csv", per_class_rows)
    write_csv(out / "tables/synthetic_predictions.csv", pred_rows)
    write_csv(out / "tables/confusion_matrices.csv", cm_rows)
    write_csv(out / "tables/optimization_trace.csv", trace)
    write_json(out / "checks/optimizer_parity.json", parity)
    write_json(out / "checks/group_separation.json", grouping)

    pilot = read_json(out / "work/pilot/pilot_manifest.json")
    assert [
        pilot[k]
        for k in (
            "session_count",
            "raw_row_count",
            "full_1hz_bin_count",
            "partial_final_raw_sample_count",
            "unlabeled_window_count",
        )
    ] == [11, 21887, 2184, 47, 94]
    qc = read_csv(out / "work/pilot/qc_sessions.csv")
    stats, sessions, status_counts = [], [], []
    raw_data = {}
    for q in qc:
        rows = read_csv(raw / q["source_file"])
        sid = q["session_id"]
        values = {c: np.array([float(r[c]) for r in rows]) for c in CHANNELS}
        times = np.array([int(r["elapsed_ms"]) / 1000 for r in rows])
        raw_data[sid] = {"channels": values, "time": times, "scenario": q["scenario"]}
        sessions.append(
            {
                "session_id": sid,
                "scenario": q["scenario"],
                "n_raw": len(rows),
                "first_time_s": float(times[0]),
                "last_time_s": float(times[-1]),
                "elapsed_span_s": float(times[-1] - times[0]),
                "interval_min_ms": int(q["interval_min_ms"]),
                "interval_max_ms": int(q["interval_max_ms"]),
                "complete_1hz_bins": int(q["full_1hz_bin_count"]),
                "partial_raw_samples": int(q["partial_final_raw_sample_count"]),
                "windows": int(q["window_count"]),
                "qc_status": q["qc_status"],
                "correlation_included": q["correlation_included"],
                "label_status": "UNLABELED",
            }
        )
        for c, v in values.items():
            q25, q75 = np.quantile(v, [0.25, 0.75], method="linear")
            stats.append(
                {
                    "session_id": sid,
                    "scenario": q["scenario"],
                    "channel": c,
                    "n_raw": len(v),
                    "median": float(np.median(v)),
                    "q25": float(q25),
                    "q75": float(q75),
                    "iqr": float(q75 - q25),
                    "minimum": float(v.min()),
                    "maximum": float(v.max()),
                    "unit": "raw_logger_units",
                    "dataset_origin": REAL,
                }
            )
            from collections import Counter

            for status, count in sorted(
                Counter(r[c + "_status"] for r in rows).items()
            ):
                status_counts.append(
                    {"session_id": sid, "channel": c, "status": status, "count": count}
                )
    write_csv(out / "tables/real_sessions.csv", sessions)
    write_csv(out / "tables/real_channel_statistics.csv", stats)
    write_csv(out / "tables/real_channel_status_counts.csv", status_counts)
    for name in (
        "sensor_correlation_median.csv",
        "sensor_correlation_iqr.csv",
        "qc_sessions.csv",
    ):
        shutil.copyfile(out / "work/pilot" / name, out / "tables" / name)
    for table in (out / "tables").glob("*.csv"):
        table.write_text(table.read_text(), newline="\n")
    exclusions = []
    for s in sessions:
        exclusions.append(
            {
                "dataset_origin": REAL,
                "session_id": s["session_id"],
                "unit": "raw sample",
                "count": s["partial_raw_samples"],
                "reason": "incomplete final group for 1 Hz aggregation",
            }
        )
    labels = read_csv(synthetic / "labels/labels.csv")
    excluded_labels = [row for row in labels if row["label_valid"].lower() != "true"]
    matrix_manifest = read_json(synthetic / "matrix/matrix_manifest.json")
    assert len(excluded_labels) == matrix_manifest["excluded_row_count"] == 96
    assert {r["window_id"] for r in labels if r["label_valid"].lower() == "true"} == {
        r["window_id"] for r in matrix
    }
    for row in labels:
        if row["label_valid"].lower() != "true":
            exclusions.append(
                {
                    "dataset_origin": SYNTHETIC,
                    "session_id": row["session_id"],
                    "unit": "window",
                    "count": 1,
                    "reason": row["exclusion_reason"],
                }
            )
    exclusions.append(
        {
            "dataset_origin": REAL,
            "session_id": "P006",
            "unit": "session in correlation only",
            "count": 1,
            "reason": "sensor-fault test; retained in raw statistics, QC and dedicated figure",
        }
    )
    write_csv(out / "tables/exclusion_audit.csv", exclusions)

    analysis = {
        "model_metrics": metrics,
        "sessions": sessions,
        "pilot_manifest": pilot,
        "matrix_manifest": read_json(synthetic / "matrix/matrix_manifest.json"),
        "optimizer_parity": parity,
        "fit_warnings": warning_rows,
        "grouping": grouping,
        "model_sha256": MODEL_HASH,
        "source_path": str(raw),
        "repo_path": str(repo),
        "repo_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "versions": {
            n: version(n)
            for n in ("numpy", "scipy", "scikit-learn", "mord", "matplotlib")
        },
        "python_version": sys.version.split()[0],
    }
    make_figures(out, analysis, trace, raw_data)
    assert input_hashes_before == repo_input_hashes(repo)
    assert all(sha(raw / r["file_name"]) == r["sha256"] for r in source_rows)
    write_json(
        out / "checks/source_preservation.json",
        {
            "status": "PASS",
            "repo_input_sha256": input_hashes_before,
            "raw_hashes_unchanged_after_analysis": True,
            "repo_inputs_unchanged_after_analysis": True,
        },
    )
    write_json(out / "analysis.json", analysis)
    write_json(
        out / "checks/analysis_checks.json",
        {
            "source_hashes_match": True,
            "bundled_model_sha_match": True,
            "no_group_overlap": True,
            "no_final_test_used": True,
            "canonical_olr_validation_match": True,
            "manual_metric_recalculation": True,
            "identical_evaluation_rows": True,
            "repeat_comparator_predictions_equal": True,
            "real_counts_reconciled": True,
            "all_11_real_sessions_retained": True,
            "raw_not_used_for_classifier_fitting": True,
            "real_classification_metrics": "NOT_EVALUABLE_NO_GROUND_TRUTH",
        },
    )
    render_report(out, analysis)
    check_links(out)
    update_artifact_manifest(out)
    print(
        json.dumps(
            {
                "result": "ANALYSIS_COMPLETE",
                "output": str(out),
                "validation": {
                    n: {
                        "accuracy": metrics[n]["validation"]["accuracy"],
                        "macro_f1": metrics[n]["validation"]["macro_f1"],
                        "kappa": metrics[n]["validation"]["quadratic_weighted_kappa"],
                    }
                    for n in MODEL_NAMES
                },
            },
            indent=2,
        )
    )


def update_artifact_manifest(out):
    analysis = read_json(out / "analysis.json")
    source_rows = read_csv(out / "tables/source_inventory.csv")
    files = [
        p
        for folder in ("tables", "figures", "checks")
        for p in (out / folder).glob("*")
        if p.is_file()
    ]
    files += [
        out / "analysis.json",
        out / "model_specifications.json",
        out / "reproduce.py",
    ]
    files += list(out.glob("*.md"))
    files += [p for p in out.glob("*.py") if p not in files]
    write_json(
        out / "artifact_manifest.json",
        {
            "repo_commit": analysis["repo_commit"],
            "model_sha256": MODEL_HASH,
            "source_sha256": {r["file_name"]: r["sha256"] for r in source_rows},
            "output_sha256": {str(p.relative_to(out)): sha(p) for p in sorted(files)},
            "reproducibility": "Fresh output required. Numeric tables/predictions must be identical; logs and timing may vary.",
            "source_inputs_immutable": True,
        },
    )


def make_figures(out, a, trace, raw_data):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#243343",
            "text.color": "#243343",
            "axes.grid": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    chart_map = []

    def save(fig, name, title, note, kind, table):
        fig.suptitle(title, fontsize=16, fontweight="bold", x=0.06, ha="left", y=0.975)
        fig.text(0.06, 0.018, note, fontsize=9, color="#485665", va="bottom")
        fig.subplots_adjust(
            top=0.84, bottom=0.17, left=0.085, right=0.97, wspace=0.3, hspace=0.48
        )
        if kind == "time_series_small_multiples":
            fig.subplots_adjust(left=0.15, top=0.90, bottom=0.13)
        elif kind == "boxplot":
            fig.subplots_adjust(left=0.15, top=0.90, bottom=0.11)
        fig.savefig(out / "figures" / name, dpi=300)
        plt.close(fig)
        chart_map.append(
            {
                "file": name,
                "question": title,
                "chart_type": kind,
                "source": table,
                "caveat": note,
            }
        )

    m = a["model_metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label in zip(
        axes, ("accuracy", "macro_f1"), ("Accuracy (%)", "Macro F1 (%)")
    ):
        x = np.arange(3)
        for j, part in enumerate(("development", "validation")):
            vals = [100 * m[n][part][metric] for n in MODEL_NAMES]
            bars = ax.bar(
                x + (j - 0.5) * 0.33,
                vals,
                0.31,
                color=COLORS,
                alpha=0.45 if j == 0 else 1,
                edgecolor="#243343",
                linewidth=0.5,
                hatch="//" if j == 0 else None,
                label="Training (76 window)" if j == 0 else "Validation (38 window)",
            )
            for b, v in zip(bars, vals):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + 1,
                    f"{v:.1f}",
                    ha="center",
                    fontsize=9,
                )
        ax.set_xticks(x, ["OLR", "Logistic\nRegression", "Decision\nTree"])
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel(label)
        ax.legend(loc="lower left", fontsize=8, framealpha=0.95)
    save(
        fig,
        "01_perbandingan_model.png",
        "Perbandingan tiga model pada data sintetis",
        "Evaluasi sintetis — mekanik pipeline. Partisi tetap; tidak mengukur akurasi data nyata.",
        "grouped_bar",
        "tables/model_metrics.csv",
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(4)
    for j, n in enumerate(MODEL_NAMES):
        vals = [r["f1"] * 100 for r in m[n]["validation"]["per_class"]]
        ax.bar(
            x + (j - 1) * 0.24,
            vals,
            0.23,
            color=COLORS[j],
            label=n,
            edgecolor="#243343",
            linewidth=0.5,
        )
    ax.set_xticks(x, CLASSES)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("F1 per kelas (%)")
    ax.legend(loc="lower left", fontsize=9)
    support = [r["support"] for r in m["OLR"]["validation"]["per_class"]]
    save(
        fig,
        "02_f1_per_kelas.png",
        "F1 per kelas pada validation sintetis",
        f"38 window; support Safe/Monitor/Caution/Urgent = {support}. Evaluasi sintetis — mekanik pipeline.",
        "grouped_bar",
        "tables/per_class_metrics.csv",
    )
    for i, n in enumerate(MODEL_NAMES):
        fig, ax = plt.subplots(figsize=(7, 6))
        cm = np.array(m[n]["validation"]["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, int(cm.max())))
        for r in range(4):
            for c in range(4):
                ax.text(
                    c,
                    r,
                    str(cm[r, c]),
                    ha="center",
                    va="center",
                    fontsize=16,
                    color="white" if cm[r, c] > cm.max() / 2 else "#243343",
                )
        ax.set_xticks(range(4), CLASSES)
        ax.set_yticks(range(4), CLASSES)
        ax.set_xlabel("Kelas prediksi")
        ax.set_ylabel("Label sintetis sebenarnya")
        colorbar = fig.colorbar(im, ax=ax, label="Jumlah window", shrink=0.85)
        colorbar.locator = MaxNLocator(integer=True)
        colorbar.update_ticks()
        save(
            fig,
            f"03_confusion_{i + 1}.png",
            f"Confusion matrix — {n}",
            f"Validation sintetis: 38 window, 2 sesi. Accuracy = {100 * m[n]['validation']['accuracy']:.2f}%.\nEvaluasi sintetis — mekanik pipeline; bukan hasil data nyata.",
            "heatmap",
            "tables/confusion_matrices.csv",
        )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    it = [r["iteration"] for r in trace]
    axes[0].plot(it, [r["objective_normalized"] for r in trace], color=COLORS[0], lw=2)
    axes[0].set_ylabel("Objective / objective awal")
    axes[0].set_title("Objective optimisasi")
    axes[1].plot(
        it,
        [100 * r["validation_accuracy"] for r in trace],
        color=COLORS[0],
        label="Accuracy",
        lw=2,
    )
    axes[1].plot(
        it,
        [100 * r["validation_macro_f1"] for r in trace],
        color=COLORS[1],
        label="Macro F1",
        lw=2,
        ls="--",
    )
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Skor validation (%)")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("Iterasi optimisasi L-BFGS-B")
    save(
        fig,
        "04_optimisasi_olr.png",
        "Jejak optimisasi OLR pada data sintetis",
        "Development 76 window; validation 38 window. Paritas optimizer diperiksa.\nKurva ini bukan bukti bebas overfitting atau akurasi sensor nyata.",
        "line",
        "tables/optimization_trace.csv",
    )
    sessions = a["sessions"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ids = [s["session_id"] for s in sessions]
    for ax, col, label in zip(
        axes, ("n_raw", "windows"), ("Jumlah sampel raw", "Jumlah window 120 detik")
    ):
        vals = [s[col] for s in sessions]
        bars = ax.barh(
            ids, vals, color=["#C47818" if s == "P006" else "#2563A6" for s in ids]
        )
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.set_xlim(0, max(vals) * 1.2)
        for b, v in zip(bars, vals):
            ax.text(
                v + max(vals) * 0.015,
                b.get_y() + b.get_height() / 2,
                str(v),
                va="center",
                fontsize=9,
            )
    save(
        fig,
        "05_cakupan_data_nyata.png",
        "Cakupan 11 sesi sensor nyata",
        "21.887 sampel; 94 window tanpa label. P006 (oranye) tetap ditampilkan sebagai sesi gangguan sensor.",
        "horizontal_bar",
        "tables/real_sessions.csv",
    )
    corr = read_csv(out / "tables/sensor_correlation_median.csv")
    arr = np.array([[float(r[c]) for c in CHANNELS] for r in corr])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(arr, cmap="PuOr", vmin=-1, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(
                j,
                i,
                f"{arr[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(arr[i, j]) > 0.65 else "#243343",
            )
    ax.set_xticks(range(5), CHANNELS)
    ax.set_yticks(range(5), CHANNELS)
    fig.colorbar(im, ax=ax, label="Median koefisien Spearman", shrink=0.85)
    save(
        fig,
        "06_korelasi_sensor.png",
        "Korelasi deskriptif lima kanal sensor",
        "Median korelasi dalam 10 sesi operasional, bobot sesi setara; P006 dikecualikan hanya dari korelasi.\nData nyata tanpa label; korelasi tidak membuktikan akurasi, sebab-akibat, atau kelayakan sensor.",
        "heatmap",
        "tables/sensor_correlation_median.csv",
    )
    for sid, data in raw_data.items():
        fig, axes = plt.subplots(5, 1, figsize=(11, 10), sharex=True)
        for i, (ax, c) in enumerate(zip(axes, CHANNELS)):
            ax.plot(
                data["time"],
                data["channels"][c],
                color=COLORS[0] if i < 2 else COLORS[1],
                lw=0.65,
            )
            ax.set_ylabel(c + "\n(raw)")
            ax.grid(axis="y", alpha=0.15)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        axes[-1].set_xlabel("Waktu relatif tercatat (detik)")
        special = " — gangguan sensor" if sid == "P006" else ""
        save(
            fig,
            f"sensor_{sid}.png",
            f"Respons sensor {sid}{special}",
            f"Skenario: {data['scenario']}; {len(data['time']):,} sampel. Lima kanal raw; tanpa penggantian outlier.\nNama skenario adalah metadata sesi, bukan label risiko tiap window.",
            "time_series_small_multiples",
            f"raw source hash in tables/source_inventory.csv: {sid}",
        )
    fig, axes = plt.subplots(5, 1, figsize=(12, 12))
    for ax, c in zip(axes, CHANNELS):
        bp = ax.boxplot(
            [raw_data[s]["channels"][c] for s in ids],
            tick_labels=ids,
            showfliers=True,
            flierprops={"markersize": 1.5, "alpha": 0.25},
            patch_artist=True,
        )
        for p, sid in zip(bp["boxes"], ids):
            p.set_facecolor("#DCE7F0" if sid != "P006" else "#F3D7AF")
        ax.set_ylabel(c + " (raw)")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    save(
        fig,
        "07_sebaran_sensor.png",
        "Sebaran pembacaan raw per sesi dan kanal",
        "Box: kuartil 25–75%; garis: median; whisker: 1,5 IQR. Titik luar whisker tetap disertakan.\nVariasi mencakup respons eksperimen dan gangguan; tidak otomatis diartikan sebagai noise.",
        "boxplot",
        "tables/real_channel_statistics.csv",
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    labels = [
        "Data sintetis\nberlabel",
        "Fitur kapasitif\n5 fitur/window",
        "Development\nfit 3 model",
        "Validation\n38 window",
        "Metrik\n4 kelas",
    ]
    for i, label in enumerate(labels):
        x = 0.015 + i * 0.2
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.58),
                0.16,
                0.26,
                boxstyle="round,pad=.015",
                facecolor="#DFEAF4",
                edgecolor=COLORS[0],
            )
        )
        ax.text(x + 0.08, 0.71, label, ha="center", va="center", fontsize=10)
        if i < 4:
            ax.annotate(
                "",
                xy=(x + 0.195, 0.71),
                xytext=(x + 0.16, 0.71),
                arrowprops={"arrowstyle": "->", "color": "#243343"},
            )
    ax.text(
        0.015,
        0.27,
        "CSV sensor nyata → QC/timing → grafik lima kanal + statistik + korelasi",
        fontsize=13,
    )
    ax.text(
        0.015,
        0.10,
        "Data nyata tetap tanpa label; tidak masuk fitting atau evaluasi klasifikasi.",
        fontsize=10,
    )
    save(
        fig,
        "08_alur_evaluasi.png",
        "Alur evaluasi AI dan karakterisasi sensor",
        "Dua sumber bukti dilaporkan terpisah. Model aplikasi tetap OLR; pembanding hanya untuk analisis sintetis.",
        "flow_diagram",
        "analysis.json",
    )
    write_json(out / "checks/chart_map.json", chart_map)


def check_links(out):
    broken = []
    for doc in out.glob("*.md"):
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", doc.read_text()):
            if target.startswith(("http:", "https:", "#")):
                continue
            if not (doc.parent / target.split("#")[0]).exists():
                broken.append([doc.name, target])
    write_json(out / "checks/local_links.json", {"broken": broken})
    assert not broken, broken


def render_report(out, a):
    m = a["model_metrics"]
    n = a["pilot_manifest"]
    tab = "| Model | Training accuracy | Validation accuracy | Macro F1 validation | Kappa kuadratik validation |\n|---|---:|---:|---:|---:|\n"
    for name in MODEL_NAMES:
        v = m[name]["validation"]
        d = m[name]["development"]
        tab += f"| {name} | {d['accuracy'] * 100:.2f}% | {v['accuracy'] * 100:.2f}% | {v['macro_f1']:.4f} | {v['quadratic_weighted_kappa']:.4f} |\n"
    per = "| Model | Kelas | Precision | Recall | F1 | Support |\n|---|---|---:|---:|---:|---:|\n"
    for name in MODEL_NAMES:
        for r in m[name]["validation"]["per_class"]:
            per += f"| {name} | {r['class_name']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['support']} |\n"
    session_table = "| Sesi | Skenario | Raw | Rentang waktu (s) | Parsial | Window |\n|---|---|---:|---:|---:|---:|\n"
    for r in a["sessions"]:
        session_table += f"| {r['session_id']} | {r['scenario']} | {r['n_raw']} | {r['elapsed_span_s']:.1f} | {r['partial_raw_samples']} | {r['windows']} |\n"
    # Expected total is derived from measured output, never a placeholder metric.
    olr = m["OLR"]["validation"]
    correct = sum(olr["confusion_matrix"][i][i] for i in range(4))
    quality_path = out / "checks/repository_quality.json"
    quality = (
        read_json(quality_path)
        if quality_path.exists()
        else {"status": "BELUM_DICATAT"}
    )
    quality_log = out / "logs/repository-quality-full.log"
    quality_detail = (
        "[log lengkap](logs/repository-quality-full.log)"
        if quality_log.exists()
        else "quality gate dijalankan terpisah dari reproduksi analisis"
    )
    repro_path = out / "checks/reproduction.json"
    repro = (
        read_json(repro_path) if repro_path.exists() else {"status": "BELUM_DICATAT"}
    )
    warning_text = (
        json.dumps(a["fit_warnings"], ensure_ascii=False)
        if a["fit_warnings"]
        else "Tidak ada warning fitting yang tercatat."
    )
    matrix = a["matrix_manifest"]
    report = f"""# Evaluasi AI OSTOSENSE dengan data yang tersedia

## Ringkasan teknis

**Paket ini menghasilkan evaluasi klasifikasi sintetis dan karakterisasi seluruh 11 sesi sensor nyata.**
OLR bawaan memprediksi benar {correct} dari 38 window validation sintetis: accuracy {olr["accuracy"] * 100:.2f}%,
Macro F1 {olr["macro_f1"]:.4f}, dan quadratic weighted kappa {olr["quadratic_weighted_kappa"]:.4f}.
Angka tersebut mengukur kesesuaian pada fixture sintetis yang telah digunakan untuk pengembangan pipeline.
Angka tersebut **bukan akurasi model pada sensor nyata** dan tidak memenuhi evaluasi Final Test nyata.

Data nyata berisi {n["raw_row_count"]:,} sampel pada 11 sesi; seluruh hash sesuai inventaris.
Belum ada label risiko atau waktu kejadian kebocoran terverifikasi. Tidak ada pengambilan data tambahan,
label buatan pada data nyata, perubahan model aplikasi, atau perubahan source repo pada pekerjaan ini.

## Data dan definisi sebelum membaca metrik

- **Training / Development:** 76 window eligible dari 4 sesi sintetis; dipakai fitting.
- **Validation:** 38 window eligible dari 2 sesi sintetis; identitas sesi, kantong, dan sensor terpisah dari Development.
- **Empat kelas:** Safe, Monitor, Caution, Urgent; label berasal dari aturan fixture sintetis, bukan anotasi data nyata.
- **Accuracy:** jumlah prediksi yang sama dengan label dibagi jumlah window. Training accuracy adalah skor in-sample.
- **Precision:** proporsi prediksi suatu kelas yang benar. **Recall:** proporsi anggota kelas sebenarnya yang terdeteksi.
- **F1:** rata-rata harmonik precision dan recall. **Macro F1:** rata-rata F1 keempat kelas dengan bobot kelas setara.
- **Kappa kuadratik:** kesepakatan label yang memperhitungkan kebetulan dan jarak kesalahan ordinal. Nilai ini bukan persen.
- **Support:** jumlah window label sebenarnya. Window bertumpang tindih; 38 window bukan 38 eksperimen independen.

## Tiga model diuji pada partisi sintetis yang sama

{tab}

Semua model menerima lima fitur yang sama. Logistic Regression dan Decision Tree adalah pembanding nominal;
OLR memodelkan urutan kelas. Tabel ini menunjukkan hasil dari parameter tetap yang disepakati sebelum evaluasi.
Tidak dilakukan pencarian parameter atau pergantian split untuk menaikkan angka. Model aplikasi tetap OLR.
Pada Validation ini, OLR dan LR masing-masing benar 36/38, sedangkan Tree benar 37/38.
Keunggulan Tree hanya satu window (2,63 poin persentase) pada dua sesi sintetis; tidak membuktikan keunggulan pada alat nyata.
Lihat [tabel angka lengkap](tables/model_metrics.csv) dan [spesifikasi parameter](model_specifications.json).

![Perbandingan model](figures/01_perbandingan_model.png)

Bar berarsir menunjukkan training, bar penuh menunjukkan validation. Selisih keduanya hanya berlaku untuk fixture ini.
Skor tinggi atau selisih kecil belum membuktikan generalisasi ke sensor nyata maupun ketiadaan overfitting.
Validation ini merupakan fixture engineering yang telah dikenal pipeline, bukan tes independen yang baru dibuka.

{per}

Tabel per kelas memperlihatkan ketidakseimbangan support; kelas dengan sedikit window memiliki estimasi yang rapuh.
Tidak dihitung confidence interval binomial seolah-olah semua window independen.

![F1 per kelas](figures/02_f1_per_kelas.png)

Perbandingan F1 membantu melihat kelas mana yang membatasi hasil agregat. Accuracy tidak menggantikan evaluasi per kelas.
Detail training dan validation tersedia pada [tabel per kelas](tables/per_class_metrics.csv).

## Confusion matrix memperlihatkan kesalahan yang sebenarnya dihitung

Setiap matrix berisi 38 window validation yang sama. Baris adalah label sintetis sebenarnya dan kolom adalah prediksi.
Jumlah diagonal dibagi total matrix sama dengan accuracy pada tabel; support sama dengan jumlah tiap baris.

![Confusion OLR](figures/03_confusion_1.png)

Matrix OLR menggunakan forward pass model bawaan yang hash-nya cocok dengan hasil reproduksi pipeline.
Kedua kesalahannya adalah label sintetis Monitor yang diprediksi Safe; recall Monitor 6/8 atau 75%.
Kesalahan di luar diagonal adalah ketidaksesuaian terhadap label sintetis, bukan kejadian pasien.

![Confusion Logistic Regression](figures/03_confusion_2.png)

Logistic Regression memakai scaling yang di-fit hanya pada Development. Hasilnya menjadi pembanding dengan data dan label yang sama.

![Confusion Decision Tree](figures/03_confusion_3.png)

Decision Tree dibatasi kedalaman tiga dan minimum dua sampel per leaf sejak awal analisis. Hasil ini tidak mewakili konfigurasi optimal semua decision tree.

## Jejak OLR menggambarkan proses optimisasi yang direproduksi

![Jejak optimisasi](figures/04_optimisasi_olr.png)

Kurva kiri menunjukkan objective relatif terhadap nilai awal, bukan neural-network loss. Kurva kanan menunjukkan skor validation
sepanjang iterasi L-BFGS-B; skor tidak digunakan untuk memilih checkpoint. Instrumentasi dijalankan ulang dan diperiksa terhadap
parameter/probabilitas/kelas model canonical sebelum grafik dibuat. Jumlah iterasi tercatat: {a["optimizer_parity"]["optimizer_iteration_count"]}.
Lihat [trace](tables/optimization_trace.csv) dan [paritas](checks/optimizer_parity.json). {warning_text}

## Semua sesi sensor nyata tetap tercakup

{session_table}

Rentang waktu adalah selisih timestamp relatif terakhir dan pertama, bukan tanggal kalender atau waktu menuju kebocoran.
CSV memiliki kolom `elapsed_ms`, sehingga grafik terhadap waktu relatif tetap dapat dibuat. Yang belum tersedia adalah
waktu kejadian pembanding terverifikasi untuk memasangkan sinyal dengan kondisi risiko sebenarnya.
Pemeriksaan menghasilkan {n["full_1hz_bin_count"]:,} kelompok 1 Hz lengkap dan {n["unlabeled_window_count"]} window tanpa label.
Sebanyak {n["partial_final_raw_sample_count"]} sampel akhir tidak membentuk kelompok lengkap berisi 10 sampel;
sampel tersebut tetap masuk statistik raw dan grafik, tetapi tidak masuk agregasi 1 Hz.

![Cakupan data nyata](figures/05_cakupan_data_nyata.png)

Panjang sesi dan jumlah window berbeda, sehingga pooled sample count tidak boleh dianggap jumlah eksperimen independen.
P006 tetap dianalisis sebagai gangguan sensor. Tabel [QC per sesi](tables/qc_sessions.csv) menyimpan interval dan status kanal.
Seluruh sesi lulus pemeriksaan struktur dengan interval tercatat 100 ms. Ini menunjukkan konsistensi struktur logger,
belum membuktikan keakuratan clock fisik atau mutu sensor. P006 memiliki 112 kelompok 1 Hz, sehingga tidak memenuhi panjang
window 120 detik dan menghasilkan nol window; seluruh 1.126 sampel raw-nya tetap tersedia.

![Sebaran sensor](figures/07_sebaran_sensor.png)

Boxplot memakai seluruh raw pada setiap sesi: kuartil dengan interpolasi linear, median, serta titik di luar 1,5 IQR.
Tidak ada penggantian outlier. Sebaran mencakup perubahan kondisi eksperimen; tanpa kondisi referensi terverifikasi,
seluruh variasi tidak dapat disebut noise. [Statistik lengkap](tables/real_channel_statistics.csv) mencatat minimum, maksimum, median, dan IQR.
Secara visual, P007 memperlihatkan kenaikan ketiga kanal kapasitif dan penurunan kedua kanal resistif sepanjang sesi.
P003 menunjukkan kenaikan besar pada Kap_7, sedangkan rentang kanal kapasitif lain berbeda. Pola ini adalah deskripsi kurva;
tanpa penanda kejadian dan besaran fisik pembanding, hubungan dengan volume atau onset kebocoran belum terverifikasi.
Satuan adalah angka raw logger, tidak dikonversi menjadi %RH, volume terkalibrasi, atau resistansi fisik.

![Korelasi](figures/06_korelasi_sensor.png)

Korelasi Spearman dihitung dalam setiap sesi operasional, lalu diringkas dengan median agar sesi panjang tidak mendominasi.
P006 dikecualikan hanya dari ringkasan korelasi sesuai konfigurasi yang sudah ada. [IQR korelasi](tables/sensor_correlation_iqr.csv)
menyertakan variasi antarsesi. Korelasi tidak menetapkan kanal terbaik, kalibrasi valid, atau akurasi model.

### P006: gangguan sensor dipertahankan sebagai bukti

![Gangguan P006](figures/sensor_P006.png)

Grafik memperlihatkan kelima kanal P006 tanpa membuang pembacaan ekstrem. Nama skenario menandai tujuan sesi;
perubahan tajam beberapa kanal tampak sekitar detik ke-54 pada kurva. Waktu tersebut hanya pengamatan visual perubahan sinyal,
bukan timestamp kejadian gangguan yang telah dikonfirmasi.
tanpa waktu kejadian terverifikasi, grafik ini tidak dipakai menghitung sensitivitas deteksi gangguan.

### Grafik setiap sesi

Grafik berikut memuat raw lengkap, sumbu waktu relatif, dan lima kanal terpisah. Tinjau pola per sesi;
nama skenario tidak memberikan label risiko untuk setiap bagian kurva.

{chr(10).join(f"- [{s['session_id']}: {s['scenario']}](figures/sensor_{s['session_id']}.png)" for s in a["sessions"])}

## Metode, pemisahan sumber bukti, dan jejak audit

![Alur](figures/08_alur_evaluasi.png)

Data sintetis melewati generator v0.3 (seed 20260722), ekstraksi fitur, rulebook fixture, matrix, fitting, dan validation.
Scaler OLR/LR hanya melihat Development; Decision Tree memakai fitur asli. Parameter tetap: OLR alpha 1/max_iter 10000;
LR C 1/lbfgs/max_iter 10000; Tree gini/depth 3/min_leaf 2/seed 20260909. Tidak ada class weighting atau tuning validation.

Data nyata menggunakan median setiap 10 sampel lengkap, baseline provisional dari 20 detik awal, dan window 120 detik/stride 10 detik.
Baseline tersebut belum diverifikasi kering. Fitur pilot memakai delta_norm, sedangkan model memakai raw-minus-baseline;
fitur pilot tidak dimasukkan ke classifier dalam paket ini. [Audit pengeluaran data](tables/exclusion_audit.csv) dan
manifest matrix menyimpan alasan window sintetis tidak eligible: {json.dumps(matrix.get("exclusion_counts", {}), ensure_ascii=False)}.

Commit AI: `{a["repo_commit"]}`. SHA-256 model: `{MODEL_HASH}`.
Versi Python {a["python_version"]}; dependensi: {", ".join(k + " " + v for k, v in a["versions"].items())}.
[Inventaris hash input](tables/source_inventory.csv), [prediksi per window](tables/synthetic_predictions.csv),
[pemisahan kelompok](checks/group_separation.json), dan [pemeriksaan numerik](checks/analysis_checks.json) disertakan.

## Keterbatasan dan bentuk klaim yang dapat digunakan

| Pertanyaan | Status |
|---|---|
| Dapatkah model menghasilkan empat kelas dan dihitung metriknya pada fixture sintetis? | Ya, angka dan prediksi disertakan. |
| Apakah klasifikasi risiko sensor nyata sudah akurat? | Belum dapat dievaluasi: tidak tersedia ground truth. |
| Apakah ada pengukuran akurasi notifikasi atau early-warning lead time? | Tidak; waktu kejadian pembanding belum tersedia. |
| Apakah angka sintetis membuktikan target Final Test nyata? | Tidak; target tersebut tetap belum dievaluasi. |
| Apakah 317 unit/integration test sama dengan 317 eksperimen sensor? | Tidak; tes kode memeriksa perilaku implementasi. |

Tidak dibuat label dari nama skenario, urutan waktu sesi, tinggi-rendah kapasitansi, atau prediksi model sendiri.
Quality gate mencakup tes kode otomatis. Paket ini belum menguji firmware on-device atau integrasi end-to-end pada alat fisik.
Hasil sangat terbatas oleh fixture, sedikitnya sesi validation, window bertumpang tindih, dan belum adanya label nyata.

## Verifikasi dan tindakan berikutnya

- Quality gate repo: **{quality["status"]}**. Detail: {quality_detail}.
- Hasil gate yang dicatat: {quality.get("summary", "lihat status dan log quality gate terpisah")}.
- Reproduksi analisis ke direktori baru: **{repro["status"]}**.
- Perbandingan numerik: {repro.get("numeric_files_compared", "belum dicatat")} artefak identik;
  pemeriksaan independen memakai scikit-learn. Bukti disimpan dalam checks/reproduction.json setelah verifikasi.
- Perhitungan manual dari confusion matrix, identitas baris evaluasi, dan pengulangan fit pembanding diperiksa oleh script.
- Selesai: paket analisis dan tabel/grafik yang tercantum. Pemeriksaan visual dicatat terpisah dalam berkas QA.
- Belum dilakukan: evaluasi akurasi nyata, notifikasi/lead time, firmware fisik, dan integrasi aplikasi pada alat fisik.
- Perlu persetujuan: perubahan keputusan model atau publikasi eksternal; tidak diperlukan untuk membaca paket lokal ini.
- Masalah/risiko: generalisasi ke data nyata belum diketahui; baseline/kualitas sensor masih provisional.

Gunakan [bahan konsultasi](bahan-konsultasi.md) untuk menjelaskan hasil dan batasannya. Pertanyaan yang masih perlu dibahas
dengan pembimbing adalah apakah pembuktian prototipe melalui mekanik model dan karakterisasi sensor ini sudah disajikan dengan tepat,
serta bagaimana menyatakan target evaluasi nyata yang belum tercapai. Tidak ada rencana pengambilan data tambahan dalam batch ini.
"""
    (out / "laporan.md").write_text(report, encoding="utf-8")
    qa = f"""# Bahan konsultasi AI OSTOSENSE

## Penjelasan pembuka

"Kami mengevaluasi tiga model menggunakan fixture sintetis berlabel dan menguji karakteristik 11 sesi sensor nyata.
Model OLR menghasilkan accuracy validation sintetis {olr["accuracy"] * 100:.2f}%, Macro F1 {olr["macro_f1"]:.4f}, dan kappa kuadratik {olr["quadratic_weighted_kappa"]:.4f}.
Kami belum menyatakan angka itu sebagai akurasi data nyata karena CSV nyata belum memiliki label risiko dan waktu kejadian pembanding yang terverifikasi."

## Pertanyaan yang perlu siap dijawab

1. **Dari mana label berasal?** Ini menentukan apa yang sebenarnya diukur oleh accuracy.
   Jawaban: label evaluasi berasal dari rulebook dan event fixture sintetis. Nama skenario P001–P007 tidak dijadikan label tiap window.
2. **Berapa data training dan validation?** Banyak window tidak sama dengan banyak percobaan independen.
   Jawaban: 76 window dari 4 sesi untuk fitting dan 38 window dari 2 sesi untuk validation; bag/sensor/sesi tidak silang.
   Window dalam sesi bertumpang tindih, dan validation sudah dikenal saat pengembangan pipeline; hasil bukan Final Test independen.
3. **Mengapa OLR dibandingkan dengan LR dan Tree?** Pemilihan pembanding perlu alasan yang jelas.
   Jawaban: OLR memodelkan urutan empat kelas; LR dan Tree memberi pembanding sederhana pada fitur, label, dan partisi yang sama.
   Parameternya tetap sejak awal. Model dengan skor fixture tertinggi tidak otomatis menjadi model terbaik pada alat nyata.
4. **Apakah sudah bebas overfitting?** Skor training tinggi saja tidak menjawab generalisasi.
   Jawaban: belum dapat disimpulkan. Selisih training/validation dilaporkan, tetapi fixture dan jumlah sesi sangat terbatas.
5. **Apa yang dibuktikan oleh data nyata?** Penting untuk membedakan hasil alat dan hasil klasifikasi.
   Jawaban: struktur/timing logger, respons raw lima kanal, statistik, korelasi deskriptif, dan pembentukan fitur tanpa label.
   Accuracy risiko, sensitivitas kebocoran, serta lead time belum dapat dihitung secara sah.
6. **Mengapa P006 tidak ikut korelasi?** Pengeluaran data harus terlihat dan beralasan.
   Jawaban: P006 adalah uji gangguan sensor, berbeda dari sesi operasional. P006 tetap ada pada QC, statistik raw, dan grafik tersendiri.
7. **Apakah semua data dipakai?** Denominator dan data parsial memengaruhi interpretasi.
   Jawaban: seluruh 21.887 raw masuk grafik/statistik; 47 raw akhir belum membentuk kelompok lengkap dan tidak masuk agregasi 1 Hz.
   Hasil agregasi berisi 2.184 kelompok lengkap dan 94 window nyata tanpa label. Window sintetis non-eligible dicatat beserta alasannya.
8. **Apa arti grafik optimisasi?** Jangan menyamakan proses model yang berbeda.
   Jawaban: objective dan skor validation sepanjang iterasi L-BFGS-B untuk OLR; bukan epoch neural network.
   Parameter, probabilitas, dan kelas hasil instrumentasi diuji paritas terhadap implementasi canonical.
9. **Apakah target Macro F1 ≥0,70 dan kappa ≥0,61 sudah tercapai?** Target memerlukan jenis bukti yang ditetapkan.
   Jawaban: belum dapat dinyatakan tercapai, karena target berlaku pada Final Test nyata, sedangkan angka paket ini sintetis.
10. **Apakah sistem sudah berguna sebagai prototipe?** Kesimpulan harus sesuai hal yang diuji.
   Jawaban: mekanik model dapat berjalan dan karakterisasi sensor tersedia. Ketepatan risiko pada kondisi nyata belum diketahui;
   penilaian kecukupan demonstrasi prototipe dibahas dengan pembimbing, tanpa menyatakan standar minimum PKM tertentu.

## Usulan kalimat untuk laporan/presentasi

"Pada evaluasi fixture sintetis dengan pemisahan Development dan Validation berdasarkan sesi, kantong, dan sensor,
OLR memperoleh accuracy {olr["accuracy"] * 100:.2f}%, Macro F1 {olr["macro_f1"]:.4f}, dan quadratic weighted kappa {olr["quadratic_weighted_kappa"]:.4f}.
Pengujian ini menunjukkan mekanik klasifikasi pada fixture tersebut. Sebanyak 11 sesi sensor nyata dianalisis secara deskriptif;
evaluasi klasifikasi pada data nyata belum dilakukan karena ground truth belum tersedia."

## Pertanyaan untuk pembimbing

"Dengan tidak memungkinkan pengambilan data tambahan, kami menyajikan evaluasi sintetis dan karakterisasi sensor nyata secara terpisah.
Apakah pemisahan hasil, keterbatasan, dan bentuk kesimpulan ini sudah tepat untuk mempertanggungjawabkan prototipe kami?"

Rujukan angka: [laporan](laporan.md) dan [metrik](tables/model_metrics.csv).
"""
    (out / "bahan-konsultasi.md").write_text(qa, encoding="utf-8")
    readme = f"""# Paket evaluasi OSTOSENSE — 9 September 2026

Buka [laporan](laporan.md), [bahan konsultasi](bahan-konsultasi.md), dan folder figures.
Paket ini bersifat lokal. Model yang dinilai adalah OLR bawaan; pembanding tidak dipasang ke aplikasi.

## Mengulang analisis

Dari terminal, jalankan menggunakan environment proyek yang sudah tersedia:

```bash
PYTHONDONTWRITEBYTECODE=1 {a["repo_path"]}/ai/.venv/bin/python {out}/reproduce.py --raw /path/to/private-csv --repo {a["repo_path"]} --output /tmp/ostosense-evaluation-new-run
```

Perintah membaca repo dan raw CSV, memeriksa hash, membentuk output di folder baru, lalu menghitung ulang tabel/grafik.
Hasil yang diharapkan: `ANALYSIS_COMPLETE`; prediksi serta metrik sama. Tujuan yang sudah memiliki analysis.json ditolak.
Gunakan `--repo` atau `--raw` hanya untuk salinan sumber yang sama, bukan data baru tanpa pembaruan protokol.
Tidak perlu instalasi paket. Data/config asal tidak berubah; tidak diperlukan rollback source.

Quality gate terpisah, dari root repo:

```bash
./scripts/verify.sh
```

Perintah memeriksa tes Python, Ruff, Pyright, build paket, kernel C++ dan hygiene; perlu loopback dan akses dependensi build.
Status analisis tidak otomatis menyatakan gate ini telah dijalankan pada komputer lain.

## Isi dan provenance

- tables: metrik full precision, prediksi per window, statistik raw, QC dan audit pengeluaran data.
- figures: PNG 300 dpi; label sintetis/nyata dan caveat tercantum pada setiap grafik.
- work: hasil turunan pipeline, bukan raw source nyata. Jangan menaruh folder ini di repository publik tanpa review.
- checks: pemeriksaan analisis, kelompok, paritas, reproduksi, dan QA.
- artifact_manifest.json: hash output numerik/grafik; analysis.json: fakta sumber laporan.

Output analisis berupa Markdown + PNG. Publikasi paket dilakukan terpisah setelah peninjauan.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--refresh-report", action="store_true")
    main(parser.parse_args())
