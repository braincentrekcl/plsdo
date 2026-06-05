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

    def test_sign_convention_largest_loading_positive(self, x_array, y_array):
        """Each component is sign-fixed so its largest-magnitude X loading is
        positive. A PLS component's global sign is arbitrary and can flip
        across BLAS builds; pinning it makes outputs reproducible across
        machines."""
        from plsdo.io import zscore_columns

        model = PLS(zscore_columns(x_array), zscore_columns(y_array))
        model.fit()

        for i in range(model.s.shape[0]):
            col = model.u_loadings[:, i]
            assert col[np.argmax(np.abs(col))] > 0

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

    def test_random_data_not_significant(self):
        """Permutation test on uncorrelated data should not yield significance."""
        rng = np.random.default_rng(99)
        X = rng.standard_normal((30, 5))
        Y = rng.standard_normal((30, 4))
        from plsdo.io import zscore_columns

        model = PLS(zscore_columns(X), zscore_columns(Y), seed=7)
        model.fit()
        model.permutation_test(n_perms=500)

        # > 0.05 is the meaningful non-significance threshold. The previous
        # 0.001 floor was vacuous: the Phipson & Smyth p-value cannot fall
        # below 1 / (n_perms + 1) ≈ 0.002, so it held even for a no-op test.
        assert np.all(model.p_values > 0.05)


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

        np.testing.assert_array_equal(m1.u_bootstrap_ratios, m2.u_bootstrap_ratios)

class TestBootstrapZscoreX:
    def test_zscore_x_false_does_not_alter_dummy_x(self):
        """Bootstrap with zscore_x=False must leave integer dummy X unchanged."""
        rng = np.random.default_rng(0)
        # Binary dummy matrix (discriminatory-style X)
        X = rng.integers(0, 2, size=(20, 3)).astype(float)
        Y = rng.standard_normal((20, 4))
        from plsdo.io import zscore_columns

        Y = zscore_columns(Y)

        model_no_zx = PLS(X.copy(), Y.copy(), seed=42, zscore_x=False)
        model_no_zx.fit()
        model_no_zx.bootstrap(n_bootstraps=50)

        # X stored on the model must be unchanged (no in-place mutation)
        np.testing.assert_array_equal(model_no_zx.X, X)

    def test_zscore_x_true_and_false_differ(self):
        """Bootstrap ratios should differ when zscore_x differs."""
        rng = np.random.default_rng(1)
        X = rng.integers(0, 2, size=(20, 3)).astype(float)
        from plsdo.io import zscore_columns

        Y = zscore_columns(rng.standard_normal((20, 4)))

        m_true = PLS(X.copy(), Y.copy(), seed=42, zscore_x=True)
        m_true.fit()
        m_true.bootstrap(n_bootstraps=50)

        m_false = PLS(X.copy(), Y.copy(), seed=42, zscore_x=False)
        m_false.fit()
        m_false.bootstrap(n_bootstraps=50)

        assert not np.allclose(m_true.u_bootstrap_ratios, m_false.u_bootstrap_ratios)


class TestFilterLVs:
    def test_before_permutation_raises(self, x_array, y_array):
        from plsdo.io import zscore_columns

        model = PLS(zscore_columns(x_array), zscore_columns(y_array))
        model.fit()
        with pytest.raises(RuntimeError, match="permutation"):
            model.filter_lvs()

    def test_before_bootstrap_raises(self, x_array, y_array):
        from plsdo.io import zscore_columns

        model = PLS(zscore_columns(x_array), zscore_columns(y_array), seed=42)
        model.fit()
        model.permutation_test(n_perms=100)
        with pytest.raises(RuntimeError, match="bootstrap"):
            model.filter_lvs()

    def test_filters_on_significance_and_reliability(self):
        """Manually set up a model with known p-values and bootstrap ratios."""
        X = np.random.default_rng(0).standard_normal((20, 3))
        Y = np.random.default_rng(1).standard_normal((20, 3))
        model = PLS(X, Y, seed=42)
        model.fit()

        # Manually set permutation results: LV1 significant, LV2 not, LV3 significant
        model.p_values = np.array([0.01, 0.50, 0.03])
        model.significant_lvs = model.p_values < 0.05
        model._permuted = True

        # Manually set bootstrap ratios:
        # LV1: reliable on both sides (|BSR| > 1.96)
        # LV3: reliable on X but not Y
        model.u_bootstrap_ratios = np.array(
            [
                [3.0, 0.5, 2.5],  # feature 1
                [0.1, 0.1, 0.1],  # feature 2
                [2.1, 0.3, 2.0],  # feature 3
            ]
        )
        model.vt_bootstrap_ratios = np.array(
            [
                [2.5, 0.2, 0.5],  # LV1: reliable
                [0.1, 0.1, 0.1],  # LV2: not reliable
                [1.0, 0.3, 1.5],  # LV3: not reliable (no feature > 1.96)
            ]
        )
        model._bootstrapped = True

        model.filter_lvs()

        # Only LV1 should survive (significant + reliable on both sides)
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(model.final_lvs, expected)


class TestKnownAnswer:
    """Feed the engine data with a planted, known structure and assert it is
    recovered. The maths is the oracle: these are the automated *correctness*
    signal for the notebook→class port, and they guard the SVD construction,
    permutation sensitivity (both directions), and the bootstrap Procrustes
    alignment against silent regression.
    """

    @staticmethod
    def _planted_rank1():
        """Rank-1 planted signal.

        A single score vector ``t`` drives two X features (loadings ``a``) and
        two Y features (loadings ``b``), each with equal magnitude and mixed
        sign so that column z-scoring preserves the structure; the remaining
        features are noise. After z-scoring the cross-covariance is ≈
        ``outer(sign(a), sign(b)) · n/(n-1)``, a rank-1 matrix whose top
        singular vectors recover ``a`` and ``b``.
        """
        from plsdo.io import zscore_columns

        n = 60
        a = np.array([1.0, -1.0, 0.0, 0.0])
        b = np.array([1.0, -1.0, 0.0])
        rng = np.random.default_rng(0)
        t = rng.standard_normal(n)
        X = np.outer(t, a) + 0.05 * rng.standard_normal((n, len(a)))
        Y = np.outer(t, b) + 0.05 * rng.standard_normal((n, len(b)))
        return zscore_columns(X), zscore_columns(Y), a, b, n

    @staticmethod
    def _near_degenerate():
        """Two planted components with nearly equal singular values.

        Bootstrap resamples rotate and swap the two near-degenerate components,
        so the Procrustes alignment in ``bootstrap()`` is essential to keep the
        loadings stable. Without it the standard errors inflate and the
        bootstrap ratios collapse.
        """
        from plsdo.io import zscore_columns

        n = 80
        rng = np.random.default_rng(0)
        t1 = rng.standard_normal(n)
        t2 = rng.standard_normal(n)
        t2 = t2 - (t2 @ t1) / (t1 @ t1) * t1  # orthogonalise t2 against t1
        a1 = np.array([1.0, 1.0, 0, 0, 0, 0])
        b1 = np.array([1.0, 1.0, 0, 0])
        a2 = np.array([0, 0, 1.0, 1.0, 0, 0])
        b2 = np.array([0, 0, 1.0, 1.0])
        c2 = 0.95  # second component slightly weaker → near-degenerate
        X = np.outer(t1, a1) + c2 * np.outer(t2, a2) + 0.3 * rng.standard_normal((n, 6))
        Y = np.outer(t1, b1) + c2 * np.outer(t2, b2) + 0.3 * rng.standard_normal((n, 4))
        return zscore_columns(X), zscore_columns(Y)

    def test_top_lv_recovers_planted_directions(self):
        """Top LV aligns with the planted loadings and dominates the spectrum."""
        X, Y, a, b, n = self._planted_rank1()
        model = PLS(X, Y, seed=42)
        model.fit()

        # Top singular value matches the analytic value 2·n/(n-1): two planted
        # features each side, |correlation| ≈ 1, scaled by the n/(n-1) divisor.
        # rel=0.01 distinguishes it from the 1/n divisor (which gives ≈ 2.0).
        assert model.s[0] == pytest.approx(2 * n / (n - 1), rel=0.01)

        # Second singular value is far smaller — the signal is rank-1.
        assert model.s[1] / model.s[0] < 0.1

        # The top singular vectors align with the planted directions (up to the
        # arbitrary global sign of an SVD component).
        ahat = a / np.linalg.norm(a)
        bhat = b / np.linalg.norm(b)
        cos_u = model.u[:, 0] @ ahat
        cos_v = model.vt[0, :] @ bhat
        assert abs(cos_u) > 0.95
        assert abs(cos_v) > 0.95
        # The X and Y sides share the same global sign (joint structure of the
        # cross-covariance), so the two cosines have the same sign.
        assert cos_u * cos_v > 0

        # The two largest loadings fall on the planted features.
        assert set(np.argsort(np.abs(model.u[:, 0]))[-2:]) == {0, 1}
        assert set(np.argsort(np.abs(model.vt[0, :]))[-2:]) == {0, 1}

    def test_permutation_significant_on_planted_not_on_random(self):
        """Permutation p is small on planted data, large on random data."""
        X, Y, _, _, _ = self._planted_rank1()
        model = PLS(X, Y, seed=42)
        model.fit()
        model.permutation_test(n_perms=499)
        # A no-op permutation (perm_order = arange) would leave the observed
        # singular value in the null every time, forcing p = 1.0 here.
        assert model.p_values[0] < 0.05

        rng = np.random.default_rng(0)
        from plsdo.io import zscore_columns

        Xr = zscore_columns(rng.standard_normal((60, 4)))
        Yr = zscore_columns(rng.standard_normal((60, 3)))
        model_r = PLS(Xr, Yr, seed=7)
        model_r.fit()
        model_r.permutation_test(n_perms=499)
        assert model_r.p_values[0] > 0.05

    def test_bootstrap_ratios_reliable_on_planted_features(self):
        """Dominant planted features are reliable (|BSR| > 1.96) on both sides
        with the correct relative-sign structure."""
        X, Y, _, _, _ = self._planted_rank1()
        model = PLS(X, Y, seed=42)
        model.fit()
        model.bootstrap(n_bootstraps=500)

        # Planted features 0 and 1 are reliable on both the X and Y sides.
        assert abs(model.u_bootstrap_ratios[0, 0]) > 1.96
        assert abs(model.u_bootstrap_ratios[1, 0]) > 1.96
        assert abs(model.vt_bootstrap_ratios[0, 0]) > 1.96
        assert abs(model.vt_bootstrap_ratios[0, 1]) > 1.96
        # Noise features are not reliable.
        assert abs(model.u_bootstrap_ratios[2, 0]) < 1.96
        assert abs(model.u_bootstrap_ratios[3, 0]) < 1.96
        # The two planted features carry opposite signs (matching a = [+, -]),
        # a global-sign-invariant structural check.
        assert np.sign(model.u_bootstrap_ratios[0, 0]) != np.sign(
            model.u_bootstrap_ratios[1, 0]
        )
        assert np.sign(model.vt_bootstrap_ratios[0, 0]) != np.sign(
            model.vt_bootstrap_ratios[0, 1]
        )

    def test_procrustes_keeps_degenerate_loadings_reliable(self):
        """With near-degenerate components, Procrustes alignment keeps all
        planted features reliable on the top LV. Without it (Q = I) the
        rotating components inflate the standard errors and the ratios drop."""
        X, Y = self._near_degenerate()
        model = PLS(X, Y, seed=42)
        model.fit()
        model.bootstrap(n_bootstraps=500)

        # All four planted features (two per component) are reliable on LV1.
        assert np.all(np.abs(model.u_bootstrap_ratios[:4, 0]) > 1.96)


class TestEdgeCases:
    """SVD-path edge cases: extreme feature counts and the discriminatory
    many-group design. Assert shapes hold so the engine degrades gracefully."""

    def test_single_x_feature(self):
        """A single X feature yields one component and runs end-to-end."""
        from plsdo.io import zscore_columns

        rng = np.random.default_rng(0)
        X = zscore_columns(rng.standard_normal((20, 1)))
        Y = zscore_columns(rng.standard_normal((20, 4)))
        model = PLS(X, Y, seed=1)
        model.fit()
        model.permutation_test(n_perms=50)
        model.bootstrap(n_bootstraps=50)
        model.filter_lvs()

        assert model.s.shape == (1,)
        assert model.u.shape == (1, 1)
        assert model.vt.shape == (1, 4)
        assert model.u_bootstrap_ratios.shape == (1, 1)
        assert model.final_lvs.shape == (1,)

    def test_additive_multifactor_trailing_lv_is_inert(self):
        """An additive K-factor dummy design is rank-deficient by K-1. The
        degenerate trailing latent variable must be harmless: ~zero singular
        value, non-significant, dropped by filter_lvs, with ~zero loadings.
        (Guards the design without switching to contrast coding.)"""
        from plsdo.io import zscore_columns

        a = np.repeat(np.arange(3), 4)  # factor A, 3 levels, 12 subjects
        b = np.tile([0, 1], 6)  # factor B, 2 levels
        X = np.column_stack([np.eye(3)[a], np.eye(2)[b]]).astype(float)
        Y = zscore_columns(np.random.default_rng(0).standard_normal((12, 4)))

        model = PLS(X, Y, seed=42, zscore_x=False)
        model.fit()
        model.permutation_test(n_perms=100)
        model.bootstrap(n_bootstraps=100)
        model.filter_lvs()

        assert model.s[-1] < 1e-8
        assert not model.significant_lvs[-1]
        assert not model.final_lvs[-1]
        assert np.abs(model.u_loadings[:, -1]).max() < 1e-6

    def test_many_group_discriminatory(self):
        """Dummy-coded X with five groups gives min(n_groups, n_y) components."""
        from plsdo.io import zscore_columns

        labels = np.repeat(np.arange(5), 5)  # 5 groups, 25 subjects
        X = np.eye(5)[labels].astype(float)  # discriminatory design (not z-scored)
        Y = zscore_columns(np.random.default_rng(0).standard_normal((25, 3)))
        model = PLS(X, Y, seed=1, zscore_x=False)
        model.fit()
        model.permutation_test(n_perms=30)
        model.bootstrap(n_bootstraps=30)
        model.filter_lvs()

        n_components = min(5, 3)
        assert model.s.shape == (n_components,)
        assert model.u.shape == (5, n_components)
        assert model.vt.shape == (n_components, 3)
        assert model.final_lvs.shape == (n_components,)


class TestInvariants:
    """Cheap properties any correct PLS decomposition must satisfy. No
    reference values needed; these guard against a future ``_decompose`` (e.g.
    a SparsePLS override) silently breaking the SVD contract.
    """

    @staticmethod
    def _planted(eps, seed=0):
        from plsdo.io import zscore_columns

        n = 60
        a = np.array([1.0, -1.0, 0, 0])
        b = np.array([1.0, -1.0, 0])
        rng = np.random.default_rng(seed)
        t = rng.standard_normal(n)
        X = np.outer(t, a) + eps * rng.standard_normal((n, len(a)))
        Y = np.outer(t, b) + eps * rng.standard_normal((n, len(b)))
        return zscore_columns(X), zscore_columns(Y)

    def test_svd_reconstructs_cross_covariance(self, x_array, y_array):
        from plsdo.io import zscore_columns

        model = PLS(zscore_columns(x_array), zscore_columns(y_array))
        model.fit()
        reconstructed = model.u @ np.diag(model.s) @ model.vt
        np.testing.assert_allclose(reconstructed, model.xcorr, atol=1e-12)

    def test_singular_vectors_orthonormal(self, x_array, y_array):
        from plsdo.io import zscore_columns

        model = PLS(zscore_columns(x_array), zscore_columns(y_array))
        model.fit()
        k = model.s.shape[0]
        np.testing.assert_allclose(model.u.T @ model.u, np.eye(k), atol=1e-12)
        np.testing.assert_allclose(model.vt @ model.vt.T, np.eye(k), atol=1e-12)

    def test_bootstrap_se_shrinks_as_snr_rises(self):
        """Stronger planted signal ⇒ smaller bootstrap SE / larger BSR on the
        dominant feature."""
        X_hi, Y_hi = self._planted(eps=0.1)
        X_lo, Y_lo = self._planted(eps=0.8)

        m_hi = PLS(X_hi, Y_hi, seed=42)
        m_hi.fit()
        m_hi.bootstrap(n_bootstraps=400)

        m_lo = PLS(X_lo, Y_lo, seed=42)
        m_lo.fit()
        m_lo.bootstrap(n_bootstraps=400)

        assert m_hi.u_se[0, 0] < m_lo.u_se[0, 0]
        assert abs(m_hi.u_bootstrap_ratios[0, 0]) > abs(m_lo.u_bootstrap_ratios[0, 0])
