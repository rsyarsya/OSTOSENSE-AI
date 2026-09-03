import unittest
from typing import TYPE_CHECKING

from ostosense_ai.evaluation import evaluate_predictions

if TYPE_CHECKING:
    import numpy as np
    from mord import LogisticAT
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler
    PIPELINE_DEPENDENCIES_AVAILABLE = True
else:
    try:
        import numpy as np
        from mord import LogisticAT
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError:
        PIPELINE_DEPENDENCIES_AVAILABLE = False
    else:
        PIPELINE_DEPENDENCIES_AVAILABLE = True


@unittest.skipUnless(
    PIPELINE_DEPENDENCIES_AVAILABLE,
    'install the optional "pipeline" dependencies to run these tests',
)
class PipelineDependencyCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(20260722)
        cls.features = rng.normal(size=(240, 5))
        latent_score = (
            1.4 * cls.features[:, 0]
            + 0.8 * cls.features[:, 1]
            - 0.5 * cls.features[:, 2]
            + rng.normal(scale=0.45, size=240)
        )
        cls.labels = np.digitize(latent_score, [-1.0, 0.0, 1.0])
        cls.groups = np.repeat(np.arange(24), 10)

        split = GroupShuffleSplit(
            n_splits=1,
            test_size=0.25,
            random_state=42,
        )
        cls.train_indices, cls.test_indices = next(
            split.split(cls.features, cls.labels, cls.groups)
        )
        cls.scaler = StandardScaler().fit(cls.features[cls.train_indices])
        cls.train_features = cls.scaler.transform(
            cls.features[cls.train_indices]
        )
        cls.test_features = cls.scaler.transform(cls.features[cls.test_indices])

    def test_training_is_reproducible_and_parameters_are_exportable(self) -> None:
        first = LogisticAT(alpha=1.0, max_iter=10_000).fit(
            self.train_features,
            self.labels[self.train_indices],
        )
        second = LogisticAT(alpha=1.0, max_iter=10_000).fit(
            self.train_features,
            self.labels[self.train_indices],
        )

        probabilities = first.predict_proba(self.test_features)
        self.assertEqual(first.coef_.shape, (5,))
        self.assertEqual(first.theta_.shape, (3,))
        self.assertTrue(np.all(np.diff(first.theta_) >= 0.0))
        self.assertTrue(np.all(np.isfinite(probabilities)))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(first.coef_, second.coef_)
        np.testing.assert_allclose(first.theta_, second.theta_)
        np.testing.assert_array_equal(
            first.predict(self.test_features),
            second.predict(self.test_features),
        )

    def test_grouped_split_and_metrics_match_golden_values(self) -> None:
        self.assertTrue(
            set(self.groups[self.train_indices]).isdisjoint(
                self.groups[self.test_indices]
            )
        )
        model = LogisticAT(alpha=1.0, max_iter=10_000).fit(
            self.train_features,
            self.labels[self.train_indices],
        )
        predictions = model.predict(self.test_features)

        # Route metrics through the canonical evaluator. This is a dependency
        # and pipeline-compatibility check only, not an OSTOSENSE performance
        # result; the golden values below are fixed for this synthetic fixture.
        result = evaluate_predictions(self.labels[self.test_indices], predictions)

        self.assertEqual(
            result["confusion_matrix"],
            [[15, 3, 0, 0], [2, 10, 1, 0], [0, 3, 14, 2], [0, 0, 1, 9]],
        )
        self.assertAlmostEqual(result["macro_f1"], 0.8009852216748768)
        self.assertAlmostEqual(result["quadratic_weighted_kappa"], 0.9138549892318737)


if __name__ == "__main__":
    unittest.main()
