"""Core PLS computation: SVD, permutation testing, bootstrap reliability."""

import numpy as np


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
