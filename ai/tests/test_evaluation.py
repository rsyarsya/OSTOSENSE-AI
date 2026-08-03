import importlib.util
import json
import math
import unittest
from unittest import mock

from ostosense_ai import evaluation
from ostosense_ai.evaluation import CLASS_NAMES, evaluate_predictions

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


class EvaluationValidationTests(unittest.TestCase):
    """Input-contract tests; these run without the optional pipeline deps."""

    def test_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_predictions([], [])

    def test_rejects_unequal_lengths(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_predictions([0, 1], [0])

    def test_rejects_labels_outside_range(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_predictions([0, 1], [0, 4])

    def test_rejects_negative_labels(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_predictions([0, 1], [0, -1])

    def test_rejects_boolean_labels(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_predictions([True, False], [0, 1])
        with self.assertRaises(ValueError):
            evaluate_predictions([0, 1], [True, False])

    def test_rejects_string_labels(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_predictions(["0", "1"], [0, 1])
        with self.assertRaises(ValueError):
            evaluate_predictions("01", [0, 1])  # a bare string is not a label sequence

    def test_rejects_float_labels(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_predictions([0.0, 1.0], [0, 1])
        with self.assertRaises(ValueError):
            evaluate_predictions([0, 1], [2.0, 3.0])

    def test_rejects_undefined_kappa(self) -> None:
        # both arrays a single identical class -> quadratic kappa is undefined
        with self.assertRaises(ValueError):
            evaluate_predictions([0, 0, 0], [0, 0, 0])

    def test_missing_sklearn_raises_actionable_runtime_error(self) -> None:
        # A valid, non-degenerate request must raise the documented RuntimeError
        # when scikit-learn is unavailable. Forcing find_spec to None makes this
        # deterministic regardless of whether sklearn is installed here.
        with mock.patch("importlib.util.find_spec", return_value=None):
            with self.assertRaises(RuntimeError) as context:
                evaluate_predictions([0, 1, 2, 3], [0, 1, 2, 3])
        message = str(context.exception)
        self.assertIn("scikit-learn", message)
        self.assertIn("pipeline", message.lower())


@unittest.skipUnless(
    SKLEARN_AVAILABLE,
    'install the optional "pipeline" dependencies to run these tests',
)
class EvaluationMetricTests(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        y_true = [0, 1, 2, 3, 0, 1, 2, 3]
        result = evaluate_predictions(y_true, list(y_true))
        self.assertEqual(
            result["confusion_matrix"],
            [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]],
        )
        self.assertEqual([c["f1"] for c in result["per_class"]], [1.0, 1.0, 1.0, 1.0])
        self.assertEqual([c["support"] for c in result["per_class"]], [2, 2, 2, 2])
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["quadratic_weighted_kappa"], 1.0)

    def test_mixed_error_golden_fixture(self) -> None:
        # Confusion (rows=truth, cols=pred):
        #   0: [3,0,0,0]   1: [1,2,0,0]   2: [0,0,2,1]   3: [0,0,0,3]
        # Per-class F1 (hand-derived): 6/7, 4/5, 4/5, 6/7 -> Macro F1 = 29/35.
        # Quadratic kappa: 1 - (num=2)/(den=34) = 16/17. See report for the working.
        y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        y_pred = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3]
        result = evaluate_predictions(y_true, y_pred)

        self.assertEqual(
            result["confusion_matrix"],
            [[3, 0, 0, 0], [1, 2, 0, 0], [0, 0, 2, 1], [0, 0, 0, 3]],
        )
        self.assertEqual([c["support"] for c in result["per_class"]], [3, 3, 3, 3])
        self.assertAlmostEqual(result["macro_f1"], 29 / 35)
        self.assertAlmostEqual(result["quadratic_weighted_kappa"], 16 / 17)

    def test_fixed_class_order_with_absent_class(self) -> None:
        # Only classes 0 and 1 are observed; 2 and 3 are absent.
        result = evaluate_predictions([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(len(result["confusion_matrix"]), 4)
        self.assertTrue(all(len(row) == 4 for row in result["confusion_matrix"]))
        self.assertEqual([c["class_name"] for c in result["per_class"]], list(CLASS_NAMES))
        caution, urgent = result["per_class"][2], result["per_class"][3]
        self.assertEqual(caution["f1"], 0.0)
        self.assertEqual(caution["support"], 0)
        self.assertEqual(urgent["f1"], 0.0)
        self.assertEqual(urgent["support"], 0)

    def test_distant_errors_penalized_more_than_adjacent(self) -> None:
        # Comparable raw correctness (both fully wrong), differing only in distance.
        adjacent = evaluate_predictions([0, 3], [1, 2])["quadratic_weighted_kappa"]
        distant = evaluate_predictions([0, 3], [3, 0])["quadratic_weighted_kappa"]
        self.assertGreater(adjacent, distant)

    def test_numpy_integer_arrays_accepted(self) -> None:
        import numpy as np

        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
        y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
        before_true = y_true.copy()
        before_pred = y_pred.copy()
        result = evaluate_predictions(y_true, y_pred)
        self.assertEqual(result["macro_f1"], 1.0)
        np.testing.assert_array_equal(y_true, before_true)  # inputs unchanged
        np.testing.assert_array_equal(y_pred, before_pred)

    def test_output_contract_is_json_finite_and_nonmutating(self) -> None:
        y_true = [0, 1, 2, 3, 1, 2]
        y_pred = [0, 1, 3, 3, 1, 2]
        true_copy, pred_copy = list(y_true), list(y_pred)
        result = evaluate_predictions(y_true, y_pred)

        # JSON serializable
        json.dumps(result)
        # all metric values finite
        self.assertTrue(math.isfinite(result["macro_f1"]))
        self.assertTrue(math.isfinite(result["quadratic_weighted_kappa"]))
        for entry in result["per_class"]:
            for key in ("precision", "recall", "f1"):
                self.assertTrue(math.isfinite(entry[key]))
        # stable class / matrix ordering
        self.assertEqual([c["class_name"] for c in result["per_class"]], list(CLASS_NAMES))
        self.assertEqual([c["name"] for c in result["class_order"]], list(CLASS_NAMES))
        self.assertEqual(result["confusion_matrix_orientation"], "rows=ground_truth, columns=predicted")
        self.assertEqual(result["sample_count"], 6)
        # inputs unchanged
        self.assertEqual(y_true, true_copy)
        self.assertEqual(y_pred, pred_copy)

    def test_scope_and_warning_present(self) -> None:
        result = evaluate_predictions([0, 1, 2, 3], [0, 1, 2, 3])
        self.assertIn("pipeline", result["evaluation_scope"].lower())
        self.assertIn("not an OSTOSENSE performance claim", result["warning"])
        self.assertNotIn("pass", result)  # no project target pass/fail field


if __name__ == "__main__":
    unittest.main()
