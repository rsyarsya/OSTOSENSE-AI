"""Canonical ordinal evaluation metrics for OSTOSENSE (pipeline test only).

This module wraps scikit-learn's established statistical metrics behind one
canonical API so every pipeline stage evaluates the four-class ordinal
classifier the same way. It computes a fixed 4x4 confusion matrix, per-class
precision/recall/F1/support, Macro F1 over all four fixed classes, and the
quadratic weighted Cohen's kappa.

Fixed ordinal class mapping (never inferred from observations):

    0 = Safe, 1 = Monitor, 2 = Caution, 3 = Urgent

The confusion matrix orientation is explicit: rows are ground-truth classes,
columns are predicted classes.

Scope: this validates metric mechanics only. A returned metric is never an
OSTOSENSE performance, notification-accuracy, early-warning, sensor, or clinical
result unless the predictions come from the approved real-data evaluation
protocol. This batch does not assess project targets, so no pass/fail field is
emitted.

scikit-learn is the authoritative implementation but is imported lazily: the
module imports cleanly without the optional ``[pipeline]`` dependencies, and a
concise ``RuntimeError`` is raised only when metrics are actually requested and
scikit-learn is genuinely absent (a broken install surfaces its real error).
"""

from __future__ import annotations

import importlib.util
import numbers
from typing import Any

CLASS_INDICES = (0, 1, 2, 3)
CLASS_NAMES = ("Safe", "Monitor", "Caution", "Urgent")
CONFUSION_MATRIX_ORIENTATION = "rows=ground_truth, columns=predicted"
EVALUATION_SCOPE = "pipeline-mechanics evaluation of metric implementation only"
NO_PERFORMANCE_WARNING = (
    "This result is not an OSTOSENSE performance claim unless the predictions "
    "come from the approved real-data evaluation protocol; it validates metric "
    "mechanics only."
)

_PIPELINE_HINT = (
    'install the optional "pipeline" dependencies (scikit-learn) to compute '
    "evaluation metrics"
)


def _load_sklearn_metrics() -> tuple[Any, Any, Any, Any]:
    """Lazily import scikit-learn metrics.

    Raises a concise ``RuntimeError`` only when scikit-learn is genuinely not
    installed. If scikit-learn is present but its import fails (a broken or
    partial install, or a broken transitive dependency), the original error
    propagates unmasked.
    """
    if importlib.util.find_spec("sklearn") is None:
        raise RuntimeError(
            f"scikit-learn is required to compute evaluation metrics; {_PIPELINE_HINT}"
        )
    from sklearn.metrics import (
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    return (
        confusion_matrix,
        precision_recall_fscore_support,
        f1_score,
        cohen_kappa_score,
    )


def _coerce_label(name: str, value: Any) -> int:
    """Return an int class label, rejecting booleans, floats, strings, and NaN-likes."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must not contain boolean labels")
    if not isinstance(value, numbers.Integral):
        raise ValueError(
            f"{name} must contain only integer labels (got {type(value).__name__})"
        )
    label = int(value)
    if label not in CLASS_INDICES:
        raise ValueError(f"{name} labels must be within {CLASS_INDICES}; got {label}")
    return label


def _validate_labels(name: str, values: Any) -> list[int]:
    """Materialize inputs into a validated list of ints without mutating the caller."""
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of integer labels, not a string")
    try:
        materialized = list(values)
    except TypeError as error:
        raise ValueError(f"{name} must be an iterable of integer labels") from error
    return [_coerce_label(name, value) for value in materialized]


def evaluate_predictions(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Evaluate ordinal predictions against ground truth over the fixed 4 classes.

    Returns a deterministic, JSON-compatible dict with the confusion matrix,
    per-class metrics, Macro F1, and quadratic weighted Cohen's kappa. Metrics
    are returned as full-precision finite floats (not rounded here).
    """
    true_labels = _validate_labels("y_true", y_true)
    pred_labels = _validate_labels("y_pred", y_pred)

    if not true_labels:
        raise ValueError("y_true and y_pred must be non-empty")
    if len(true_labels) != len(pred_labels):
        raise ValueError(
            f"y_true and y_pred must have equal length "
            f"({len(true_labels)} != {len(pred_labels)})"
        )

    # Quadratic weighted kappa is mathematically undefined when every observed
    # label (across both arrays) is the same single class: the expected-agreement
    # denominator is zero. Reject explicitly instead of emitting NaN.
    if len({*true_labels, *pred_labels}) == 1:
        raise ValueError(
            "quadratic_weighted_kappa is undefined when all labels are a single "
            "identical class"
        )

    (
        confusion_matrix,
        precision_recall_fscore_support,
        f1_score,
        cohen_kappa_score,
    ) = _load_sklearn_metrics()

    labels = list(CLASS_INDICES)
    matrix = confusion_matrix(true_labels, pred_labels, labels=labels)
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, labels=labels, zero_division=0
    )
    macro_f1 = f1_score(
        true_labels, pred_labels, labels=labels, average="macro", zero_division=0
    )
    quadratic_weighted_kappa = cohen_kappa_score(
        true_labels, pred_labels, labels=labels, weights="quadratic"
    )

    per_class = [
        {
            "class_index": index,
            "class_name": name,
            "precision": float(precision[position]),
            "recall": float(recall[position]),
            "f1": float(per_class_f1[position]),
            "support": int(support[position]),
        }
        for position, (index, name) in enumerate(zip(CLASS_INDICES, CLASS_NAMES))
    ]

    return {
        "evaluation_scope": EVALUATION_SCOPE,
        "class_order": [
            {"index": index, "name": name}
            for index, name in zip(CLASS_INDICES, CLASS_NAMES)
        ],
        "sample_count": len(true_labels),
        "confusion_matrix": [[int(cell) for cell in row] for row in matrix],
        "confusion_matrix_orientation": CONFUSION_MATRIX_ORIENTATION,
        "per_class": per_class,
        "macro_f1": float(macro_f1),
        "quadratic_weighted_kappa": float(quadratic_weighted_kappa),
        "warning": NO_PERFORMANCE_WARNING,
    }
