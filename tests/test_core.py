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
