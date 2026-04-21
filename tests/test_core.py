import numpy as np
import pytest
from plsdo.core import PLS


class TestPLSFit:
    def test_fit_stores_results(self, x_array, y_array):
        from plsdo.io import zscore_columns

        X = zscore_columns(x_array)
        Y = zscore_columns(y_array)
        model = PLS(X, Y)
        model.fit()

        n_subjects, n_x = X.shape
        n_y = Y.shape[1]
        n_components = min(n_x, n_y)

        assert model.xcorr.shape == (n_x, n_y)
        assert model.u.shape == (n_x, n_components)
        assert model.s.shape == (n_components,)
        assert model.vt.shape == (n_components, n_y)
        assert model.u_loadings.shape == (n_x, n_components)
        assert model.vt_loadings.shape == (n_components, n_y)
        assert model.x_scores.shape == (n_subjects, n_components)
        assert model.y_scores.shape == (n_subjects, n_components)

    def test_cross_correlation_formula(self, x_array, y_array):
        from plsdo.io import zscore_columns

        X = zscore_columns(x_array)
        Y = zscore_columns(y_array)
        model = PLS(X, Y)
        model.fit()

        expected = X.T @ Y / (X.shape[0] - 1)
        np.testing.assert_allclose(model.xcorr, expected)

    def test_loadings_are_scaled_vectors(self, x_array, y_array):
        from plsdo.io import zscore_columns

        X = zscore_columns(x_array)
        Y = zscore_columns(y_array)
        model = PLS(X, Y)
        model.fit()

        expected_u_load = model.u @ np.diag(model.s)
        expected_vt_load = np.diag(model.s) @ model.vt
        np.testing.assert_allclose(model.u_loadings, expected_u_load)
        np.testing.assert_allclose(model.vt_loadings, expected_vt_load)

    def test_scores_are_projections(self, x_array, y_array):
        from plsdo.io import zscore_columns

        X = zscore_columns(x_array)
        Y = zscore_columns(y_array)
        model = PLS(X, Y)
        model.fit()

        np.testing.assert_allclose(model.x_scores, X @ model.u)
        np.testing.assert_allclose(model.y_scores, Y @ model.vt.T)

    def test_singular_values_descending(self, x_array, y_array):
        from plsdo.io import zscore_columns

        X = zscore_columns(x_array)
        Y = zscore_columns(y_array)
        model = PLS(X, Y)
        model.fit()

        assert all(
            model.s[i] >= model.s[i + 1] for i in range(len(model.s) - 1)
        )


class TestPermutationTest:
    def _fitted_model(self, x_array, y_array):
        from plsdo.io import zscore_columns

        X = zscore_columns(x_array)
        Y = zscore_columns(y_array)
        model = PLS(X, Y, seed=42)
        model.fit()
        return model

    def test_before_fit_raises(self, x_array, y_array):
        from plsdo.io import zscore_columns

        model = PLS(zscore_columns(x_array), zscore_columns(y_array))
        with pytest.raises(RuntimeError, match="fit"):
            model.permutation_test()

    def test_stores_results(self, x_array, y_array):
        model = self._fitted_model(x_array, y_array)
        model.permutation_test(n_perms=100)

        n_components = min(x_array.shape[1], y_array.shape[1])
        assert model.p_values.shape == (n_components,)
        assert model.permuted_singular_values.shape == (n_components, 100)
        assert model.significant_lvs.dtype == bool

    def test_p_values_between_0_and_1(self, x_array, y_array):
        model = self._fitted_model(x_array, y_array)
        model.permutation_test(n_perms=100)
        assert np.all(model.p_values >= 0.0)
        assert np.all(model.p_values <= 1.0)

    def test_seed_reproducibility(self, x_array, y_array):
        from plsdo.io import zscore_columns

        X = zscore_columns(x_array)
        Y = zscore_columns(y_array)

        m1 = PLS(X, Y, seed=42)
        m1.fit()
        m1.permutation_test(n_perms=100)

        m2 = PLS(X, Y, seed=42)
        m2.fit()
        m2.permutation_test(n_perms=100)

        np.testing.assert_array_equal(m1.p_values, m2.p_values)
        np.testing.assert_array_equal(
            m1.permuted_singular_values, m2.permuted_singular_values
        )


class TestBootstrap:
    def _fitted_model(self, x_array, y_array):
        from plsdo.io import zscore_columns

        X = zscore_columns(x_array)
        Y = zscore_columns(y_array)
        model = PLS(X, Y, seed=42)
        model.fit()
        return model

    def test_before_fit_raises(self, x_array, y_array):
        from plsdo.io import zscore_columns

        model = PLS(zscore_columns(x_array), zscore_columns(y_array))
        with pytest.raises(RuntimeError, match="fit"):
            model.bootstrap()

    def test_stores_results(self, x_array, y_array):
        model = self._fitted_model(x_array, y_array)
        model.bootstrap(n_bootstraps=100)

        n_x = x_array.shape[1]
        n_y = y_array.shape[1]
        n_components = min(n_x, n_y)

        assert model.u_bootstrap_ratios.shape == (n_x, n_components)
        assert model.vt_bootstrap_ratios.shape == (n_components, n_y)
        assert model.u_se.shape == (n_x, n_components)
        assert model.vt_se.shape == (n_components, n_y)

    def test_bootstrap_ratios_are_loadings_over_se(self, x_array, y_array):
        model = self._fitted_model(x_array, y_array)
        model.bootstrap(n_bootstraps=100)

        eps = 1e-12
        expected_u_bsr = model.u_loadings / np.maximum(model.u_se, eps)
        expected_vt_bsr = model.vt_loadings / np.maximum(model.vt_se, eps)
        np.testing.assert_allclose(model.u_bootstrap_ratios, expected_u_bsr)
        np.testing.assert_allclose(model.vt_bootstrap_ratios, expected_vt_bsr)

    def test_seed_reproducibility(self, x_array, y_array):
        from plsdo.io import zscore_columns

        X = zscore_columns(x_array)
        Y = zscore_columns(y_array)

        m1 = PLS(X, Y, seed=42)
        m1.fit()
        m1.bootstrap(n_bootstraps=100)

        m2 = PLS(X, Y, seed=42)
        m2.fit()
        m2.bootstrap(n_bootstraps=100)

        np.testing.assert_array_equal(
            m1.u_bootstrap_ratios, m2.u_bootstrap_ratios
        )
