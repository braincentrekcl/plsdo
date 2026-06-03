"""Shared statistical helpers."""

import numpy as np


def corrected_pvalue(observed, null, axis: int = -1):
    """Permutation p-value with the Phipson & Smyth (2010) +1 correction.

    Computes ``(#{null >= observed} + 1) / (n_permutations + 1)``. The +1 in
    numerator and denominator keeps the p-value strictly positive — a
    permutation test can never legitimately return p = 0 — and gives exact
    Type I error control.

    Parameters
    ----------
    observed : float or ndarray
        Observed statistic(s).
    null : ndarray
        Null distribution. ``axis`` holds the permutation replicates; any
        leading axes must broadcast against ``observed`` (e.g. one row of
        permuted singular values per latent variable).
    axis : int
        Axis of ``null`` holding the permutation replicates (default last).

    Returns
    -------
    float or ndarray
        Corrected p-value(s), shaped like ``observed``.
    """
    observed = np.asarray(observed)
    null = np.asarray(null)
    n_perms = null.shape[axis]
    exceed = np.sum(null >= np.expand_dims(observed, axis), axis=axis)
    return (exceed + 1) / (n_perms + 1)
