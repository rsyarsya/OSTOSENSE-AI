import copy
import importlib.util
import math
import unittest
from typing import Any, cast

from ostosense_ai import inference
from ostosense_ai.inference import InferenceError, predict_exported_model

PIPELINE_AVAILABLE = all(
    importlib.util.find_spec(m) is not None for m in ("numpy", "sklearn", "mord", "scipy")
)


def _model(**overrides: Any) -> dict[str, Any]:
    model = {
        "model_artifact_version": "0.1.0",
        "model_family": "mord.LogisticAT",
        "dataset_origin": "SYNTHETIC_PIPELINE_TEST_ONLY",
        "feature_order": list(inference.FEATURE_COLUMNS),
        "class_order": list(inference.CLASS_NAMES),
        "class_mapping": dict(inference.CLASS_NAME_TO_INDEX),
        "scaler": {"mean": [0.0, 0.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0, 1.0, 1.0]},
        "beta": [1.0, 0.0, 0.0, 0.0, 0.0],
        "theta": [-1.0, 0.0, 1.0],
    }
    model.update(overrides)
    return model


class InferenceGoldenTests(unittest.TestCase):
    def test_golden_hand_calculation(self):
        result = predict_exported_model(_model(), [[0.0, 0, 0, 0, 0], [5.0, 0, 0, 0, 0]])
        s1 = 1.0 / (1.0 + math.exp(1.0))   # sigmoid(-1)
        s3 = 1.0 / (1.0 + math.exp(-1.0))  # sigmoid(1)
        expected0 = [s1, 0.5 - s1, s3 - 0.5, 1.0 - s3]
        for got, exp in zip(result["probabilities"][0], expected0):
            self.assertAlmostEqual(got, exp, places=12)
        self.assertAlmostEqual(sum(result["probabilities"][0]), 1.0, places=12)
        # eta=0 gives a two-way tie between Safe(0) and Urgent(3) -> lowest index wins
        self.assertEqual(result["predicted_indices"][0], 0)
        self.assertEqual(result["predicted_labels"][0], "Safe")
        # eta=5 pushes mass to the top class
        self.assertEqual(result["predicted_indices"][1], 3)
        self.assertEqual(result["predicted_labels"][1], "Urgent")

    def test_probabilities_finite_and_unit_sum(self):
        result = predict_exported_model(_model(), [[x, -x, 0.5 * x, 0.0, 0.0] for x in (-3, 0, 3)])
        for row in result["probabilities"]:
            self.assertTrue(all(math.isfinite(p) and -1e-9 <= p <= 1 + 1e-9 for p in row))
            self.assertAlmostEqual(sum(row), 1.0, places=12)


class InferenceValidationTests(unittest.TestCase):
    def test_malformed_model_artifact(self):
        mutations = [
            {"model_artifact_version": "9.9.9"},
            {"model_family": "sklearn.LogisticRegression"},
            {"dataset_origin": "REAL"},
            {"feature_order": ["a", "b", "c", "d", "e"]},
            {"class_order": ["Urgent", "Caution", "Monitor", "Safe"]},
            {"class_mapping": {"Safe": 3, "Monitor": 2, "Caution": 1, "Urgent": 0}},
            {"scaler": {"mean": [0, 0, 0, 0], "scale": [1, 1, 1, 1, 1]}},   # bad mean length
            {"scaler": {"mean": [0, 0, 0, 0, 0], "scale": [1, 1, 0.0, 1, 1]}},  # non-positive scale
            {"beta": [1, 2, 3]},                                            # bad beta length
            {"theta": [0.0, 1.0]},                                          # bad theta length
            {"theta": [1.0, 0.0, -1.0]},                                    # not increasing
            {"beta": [1.0, float("nan"), 0.0, 0.0, 0.0]},                   # non-finite
            {"theta": [-1.0, float("inf"), 1.0]},                           # non-finite
        ]
        for overrides in mutations:
            with self.assertRaises(InferenceError):
                predict_exported_model(_model(**overrides), [[0, 0, 0, 0, 0]])

    def test_bad_feature_rows(self):
        with self.assertRaises(InferenceError):
            predict_exported_model(_model(), [[0, 0, 0, 0]])          # wrong width
        with self.assertRaises(InferenceError):
            predict_exported_model(_model(), [[0, 0, "x", 0, 0]])     # non-numeric
        with self.assertRaises(InferenceError):
            predict_exported_model(_model(), [[0, float("nan"), 0, 0, 0]])  # non-finite
        with self.assertRaises(InferenceError):
            predict_exported_model(_model(), cast(Any, [None]))            # non-iterable row

    def test_inputs_not_mutated(self):
        model = _model()
        snapshot = copy.deepcopy(model)
        rows = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        predict_exported_model(model, rows)
        self.assertEqual(model, snapshot)
        self.assertEqual(rows, [[1.0, 2.0, 3.0, 4.0, 5.0]])


@unittest.skipUnless(PIPELINE_AVAILABLE, "requires the optional [pipeline] dependencies")
class InferenceMordParityTests(unittest.TestCase):
    def test_matches_mord_probabilities_and_labels(self):
        import numpy as np
        from mord import LogisticAT
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(20260722)
        features = np.vstack([rng.normal(c, 0.3, size=(8, 5)) for c in range(4)])
        targets = np.repeat(np.arange(4), 8)
        scaler = StandardScaler().fit(features)
        scaled = scaler.transform(features)
        model = LogisticAT(alpha=1.0, max_iter=10_000).fit(scaled, targets)
        mean = scaler.mean_
        scale = scaler.scale_
        assert mean is not None
        assert scale is not None

        artifact = _model(
            scaler={
                "mean": [round(v, 12) for v in np.asarray(mean).ravel()],
                "scale": [round(v, 12) for v in np.asarray(scale).ravel()],
            },
            beta=[round(v, 12) for v in np.asarray(model.coef_).ravel()],
            theta=[round(v, 12) for v in np.asarray(model.theta_).ravel()],
        )
        result = predict_exported_model(artifact, features.tolist())
        mord_proba = model.predict_proba(scaled)
        mord_pred = model.predict(scaled)
        max_diff = float(np.max(np.abs(np.asarray(result["probabilities"]) - mord_proba)))
        self.assertLessEqual(max_diff, 1e-6)
        self.assertEqual(result["predicted_indices"], list(mord_pred))


if __name__ == "__main__":
    unittest.main()
