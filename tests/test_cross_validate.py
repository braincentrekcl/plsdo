import numpy as np
from plsdo.cross_validate import run_cv, permutation_test_cv


class TestRunCV:
    def test_perfect_separation(self):
        """Groups with zero overlap should classify near-perfectly."""
        rng = np.random.default_rng(42)
        n_per_group = 20
        # Group 0: features centred at 0, Group 1: at 10
        X = np.vstack(
            [
                rng.standard_normal((n_per_group, 5)),
                rng.standard_normal((n_per_group, 5)) + 10,
            ]
        )
        labels = np.array([0] * n_per_group + [1] * n_per_group)

        results = run_cv(X, labels, n_splits=5, n_repeats=10, n_components=1, seed=42)
        assert results["mean_accuracy"] > 0.90

    def test_random_data_near_chance(self):
        """Random labels should give accuracy near chance (0.5 for 2 groups)."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 5))
        labels = np.array([0] * 20 + [1] * 20)
        rng.shuffle(labels)

        results = run_cv(X, labels, n_splits=5, n_repeats=10, n_components=1, seed=42)
        assert results["mean_accuracy"] < 0.70  # generous margin

    def test_seed_reproducibility(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 5))
        labels = np.array([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10)

        r1 = run_cv(X, labels, n_splits=5, n_repeats=10, n_components=3, seed=42)
        r2 = run_cv(X, labels, n_splits=5, n_repeats=10, n_components=3, seed=42)
        assert r1["mean_accuracy"] == r2["mean_accuracy"]

    def test_returns_expected_keys(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 5))
        labels = np.array([0] * 20 + [1] * 20)

        results = run_cv(X, labels, n_splits=5, n_repeats=2, n_components=1, seed=42)
        assert "mean_accuracy" in results
        assert "mean_balanced_accuracy" in results
        assert "fold_results" in results
        assert "true_labels" in results
        assert "pred_labels" in results
        assert "confusion_matrix" in results


class TestPermutationTestCV:
    def test_returns_p_value(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 5))
        labels = np.array([0] * 20 + [1] * 20)

        result = permutation_test_cv(
            X,
            labels,
            observed_accuracy=0.5,
            n_splits=5,
            n_repeats=2,
            n_components=1,
            n_permutations=50,
            seed=42,
        )
        assert 0.0 <= result["p_value"] <= 1.0
        assert "null_accuracies" in result
        assert len(result["null_accuracies"]) == 50
