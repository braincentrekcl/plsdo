import numpy as np
import pytest
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

    def test_p_value_at_least_one_over_n_plus_one(self):
        """Phipson-Smyth correction: p can never be 0, even for a huge observed."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 5))
        labels = np.array([0] * 20 + [1] * 20)

        result = permutation_test_cv(
            X,
            labels,
            observed_accuracy=1.0,  # nothing in the null can exceed this
            n_splits=5,
            n_repeats=1,
            n_components=1,
            n_permutations=20,
            seed=42,
        )
        assert result["p_value"] == pytest.approx(1 / (20 + 1))


class TestMulticlass:
    def test_confusion_matrix_shape_and_rows_normalised(self):
        """Three groups -> 3x3 matrix; rows are 'true'-normalised so sum to 1."""
        rng = np.random.default_rng(1)
        X = np.vstack(
            [
                rng.standard_normal((12, 5)),
                rng.standard_normal((12, 5)) + 8,
                rng.standard_normal((12, 5)) - 8,
            ]
        )
        labels = np.array([0] * 12 + [1] * 12 + [2] * 12)

        results = run_cv(X, labels, n_splits=3, n_repeats=4, n_components=2, seed=42)
        cm = results["confusion_matrix"]
        assert cm.shape == (3, 3)
        np.testing.assert_allclose(cm.sum(axis=1), np.ones(3))

    def test_predicted_labels_within_group_range(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((30, 5))
        labels = np.array([0] * 10 + [1] * 10 + [2] * 10)

        results = run_cv(X, labels, n_splits=5, n_repeats=2, n_components=2, seed=42)
        assert set(results["pred_labels"]).issubset({0, 1, 2})


class TestCVEdgeCases:
    def test_too_few_subjects_raises_clear_error(self):
        """Fewer subjects than folds must fail loudly, not silently."""
        X = np.random.default_rng(0).standard_normal((4, 5))
        labels = np.array([0, 0, 1, 1])
        with pytest.raises(ValueError, match="n_splits"):
            run_cv(X, labels, n_splits=5, n_repeats=1, n_components=1, seed=0)


class TestSklearnImportGuard:
    def test_missing_sklearn_raises_helpful_error(self, monkeypatch):
        """Importing cross_validate without scikit-learn points at plsdo[cv]."""
        import builtins
        import importlib
        import sys

        for mod in list(sys.modules):
            if mod == "sklearn" or mod.startswith("sklearn.") or mod == "plsdo.cross_validate":
                monkeypatch.delitem(sys.modules, mod, raising=False)

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "sklearn" or name.startswith("sklearn."):
                raise ImportError("simulated missing scikit-learn")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError, match=r"plsdo\[cv\]"):
            importlib.import_module("plsdo.cross_validate")
