"""Core PLS computation: SVD, permutation testing, bootstrap reliability."""

import numpy as np
from scipy.linalg import orthogonal_procrustes

from plsdo.io import zscore_columns


class PLS:
    """Partial Least Squares via SVD of the cross-covariance matrix.

    Handles both correlational PLS (two continuous matrices) and
    discriminatory PLS (design matrix vs continuous matrix) — the
    maths is identical.

    Parameters
    ----------
    X : ndarray, shape (n_subjects, n_x_features)
        First data matrix (already z-scored for correlational,
        or dummy-coded for discriminatory).
    Y : ndarray, shape (n_subjects, n_y_features)
        Second data matrix (already z-scored).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, seed: int | None = None):
        self.X = X
        self.Y = Y
        self.n_subjects = X.shape[0]
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._fitted = False

    def fit(self):
        """Run PLS: cross-covariance, SVD, loadings, and subject scores."""
        self.xcorr = self.X.T @ self.Y / (self.n_subjects - 1)
        self._decompose()
        self.u_loadings = self.u @ np.diag(self.s)
        self.vt_loadings = np.diag(self.s) @ self.vt
        self.x_scores = self.X @ self.u
        self.y_scores = self.Y @ self.vt.T
        self._fitted = True

    def _decompose(self):
        """SVD of the cross-covariance matrix.

        Separated into its own method so that SparsePLS can override
        just this step.
        """
        self.u, self.s, self.vt = np.linalg.svd(
            self.xcorr, full_matrices=False
        )

    def _check_fitted(self):
        """Raise if fit() has not been called."""
        if not self._fitted:
            raise RuntimeError(
                "Call .fit() before running permutation or bootstrap."
            )

    def permutation_test(self, n_perms: int = 10000) -> None:
        """Test significance of singular values by permutation.

        Permutes rows of X to break the subject-level pairing,
        recomputes SVD, and compares observed singular values
        to the null distribution.

        Parameters
        ----------
        n_perms : int
            Number of permutations.
        """
        self._check_fitted()

        perm_rng = np.random.default_rng(self._rng.integers(2**31))
        n_components = len(self.s)

        perm_s_list = []
        for _ in range(n_perms):
            perm_order = perm_rng.permutation(self.n_subjects)
            perm_x = self.X[perm_order, :]
            perm_xcorr = perm_x.T @ self.Y / (self.n_subjects - 1)
            _, perm_s, _ = np.linalg.svd(perm_xcorr, full_matrices=False)
            perm_s_list.append(perm_s)

        self.permuted_singular_values = np.stack(perm_s_list, axis=1)
        self.p_values = np.mean(
            self.permuted_singular_values >= self.s[:, None], axis=1
        )
        self.significant_lvs = self.p_values < 0.05

    def bootstrap(self, n_bootstraps: int = 10000) -> None:
        """Assess reliability of loadings via bootstrap resampling.

        Resamples subjects with replacement, recomputes SVD, aligns
        via Procrustes rotation, and computes bootstrap ratios
        (loadings / standard error).

        Parameters
        ----------
        n_bootstraps : int
            Number of bootstrap resamples.
        """
        self._check_fitted()

        boot_rng = np.random.default_rng(self._rng.integers(2**31))
        row_idx = np.arange(self.n_subjects)

        u_distribution = []
        vt_distribution = []

        for _ in range(n_bootstraps):
            idx = boot_rng.choice(row_idx, size=self.n_subjects, replace=True)
            x_boot = zscore_columns(self.X[idx, :])
            y_boot = zscore_columns(self.Y[idx, :])

            boot_xcorr = x_boot.T @ y_boot / (self.n_subjects - 1)
            boot_u, boot_s, boot_vt = np.linalg.svd(
                boot_xcorr, full_matrices=False
            )

            # Procrustes: rotate bootstrap Vt to align with reference
            Q, _ = orthogonal_procrustes(boot_vt.T, self.vt.T)

            boot_u_load = boot_u @ np.diag(boot_s)
            boot_vt_load = np.diag(boot_s) @ boot_vt

            aligned_u_load = boot_u_load @ Q
            aligned_vt_load = Q.T @ boot_vt_load

            # Sign correction
            signs = np.sign(
                np.sum(aligned_vt_load * self.vt_loadings, axis=1, keepdims=True)
            )
            signs[signs == 0] = 1.0

            u_distribution.append(aligned_u_load * signs.T)
            vt_distribution.append(aligned_vt_load * signs)

        self.u_se = np.std(np.stack(u_distribution, axis=2), axis=2)
        self.vt_se = np.std(np.stack(vt_distribution, axis=2), axis=2)

        eps = 1e-12
        self.u_bootstrap_ratios = self.u_loadings / np.maximum(self.u_se, eps)
        self.vt_bootstrap_ratios = self.vt_loadings / np.maximum(
            self.vt_se, eps
        )
