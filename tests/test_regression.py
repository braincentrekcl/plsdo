"""Regression snapshot guard against silent drift in pipeline outputs.

This locks the result-defining numbers produced end-to-end by the pipelines on
the committed synthetic data, so a future change that silently alters results
(e.g. the 1/(n-1) divisor, the z-scoring ddof, the bootstrap-ratio definition,
or the sign convention) is caught. It is a *drift* guard, not a correctness
oracle — correctness rests on the known-answer tests and the cited literature.

Tolerances are tiered by how reproducible each quantity is across machines:

* deterministic outputs (singular values, loadings, subject scores) are pinned
  tightly over all latent variables — they depend only on the SVD and the fixed
  sign convention, and are reproducible across BLAS builds;
* permutation p-values and bootstrap ratios are pinned only for the *significant*
  latent variables (see ``_significant_lvs``); for non-significant LVs these
  quantities are both meaningless (discarded by ``filter_lvs``) and unstable
  across BLAS builds, so pinning them gives false failures, not drift detection.
  p-values use an absolute tolerance (they are counts); bootstrap ratios use a
  looser tolerance, as they accumulate per-resample SVD differences. A genuine
  code regression moves these far more than the tolerance, so the guard still
  bites.

Regenerating the snapshot is a deliberate, reviewed step (e.g. a major
numpy/scipy bump): delete ``tests/data/regression/`` and run the suite once to
recreate it, then commit. Never regenerate to paper over an unexplained diff.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plsdo.pipeline import cross_validate_pipeline, run_pipeline

DATA_DIR = Path(__file__).parent / "data"
REF_DIR = DATA_DIR / "regression"

TIGHT = dict(rtol=1e-6, atol=1e-9)
PVAL = dict(rtol=0.0, atol=0.02)
LOOSE = dict(rtol=1e-2, atol=1e-2)
SCALAR = dict(rtol=0.0, atol=0.05)


def _load(path: Path, index_col=None) -> np.ndarray:
    return pd.read_csv(path, index_col=index_col).to_numpy(dtype=float)


def _snapshot(name: str, arr: np.ndarray, tol: dict, cols=None) -> None:
    """Compare ``arr`` against the committed snapshot, or create it if absent.

    On a fresh ``tests/data/regression/`` (deliberate regeneration) the
    snapshot is written and the assertion is skipped for that quantity; on all
    later runs the committed snapshot is asserted against. The full array is
    always saved; ``cols`` restricts only the *comparison* to a subset of
    latent-variable columns (the last axis), so the snapshot file stays
    complete and transparent while the assertion ignores quantities that are
    not reproducible across BLAS builds (see ``_significant_lvs``).
    """
    ref_path = REF_DIR / f"{name}.npy"
    arr = np.asarray(arr)
    if not ref_path.exists():
        REF_DIR.mkdir(parents=True, exist_ok=True)
        np.save(ref_path, arr)
        return
    ref = np.load(ref_path)
    if cols is not None:
        arr, ref = arr[..., cols], ref[..., cols]
    np.testing.assert_allclose(arr, ref, err_msg=f"regression drift in '{name}'", **tol)


def _significant_lvs(method: str):
    """Column mask of the latent variables significant in the reference snapshot.

    Permutation p-values and bootstrap ratios are pinned only for these LVs.
    A non-significant LV is discarded by ``filter_lvs`` and never affects a
    result, and its p-value and bootstrap ratios are also unstable across BLAS
    builds: a near-zero or weak singular value puts the observed statistic in a
    dense pile of near-equal null values, so sub-precision SVD differences flip
    many permutation counts, and a near-zero loading makes the bootstrap ratio a
    ratio of numerical noise. Non-significant LVs remain guarded by the
    deterministic singular-value, loading and score snapshots, which are
    reproducible across machines and catch the result-defining drift (the
    1/(n-1) divisor, the z-scoring ddof, the sign convention).

    Returns ``None`` during a deliberate regeneration (no committed p-values
    yet), which leaves the comparison unrestricted.
    """
    p_path = REF_DIR / f"{method}_p_values.npy"
    if not p_path.exists():
        return None
    return np.load(p_path).ravel() < 0.05


def _run_pls(method: str, out: Path) -> Path:
    kwargs = dict(
        method=method,
        y_path=DATA_DIR / "behaviour.csv",
        demographics_path=DATA_DIR / "demographics.csv",
        output_dir=out,
        group_col="group",
        subject_id="subject_id",
        n_perms=200,
        n_bootstraps=200,
        seed=42,
        img_format="png",
        dpi=72,
    )
    if method == "correlational":
        kwargs["x_path"] = DATA_DIR / "brain.csv"
    run_pipeline(**kwargs)
    return out / "data"


@pytest.fixture(scope="module")
def correlational_data(tmp_path_factory):
    return _run_pls("correlational", tmp_path_factory.mktemp("corr"))


@pytest.fixture(scope="module")
def discriminatory_data(tmp_path_factory):
    return _run_pls("discriminatory", tmp_path_factory.mktemp("disc"))


@pytest.fixture(scope="module")
def cv_data(tmp_path_factory):
    out = tmp_path_factory.mktemp("cv")
    cross_validate_pipeline(
        y_path=DATA_DIR / "behaviour.csv",
        demographics_path=DATA_DIR / "demographics.csv",
        output_dir=out,
        group_col="group",
        subject_id="subject_id",
        n_folds=3,
        n_repeats=5,
        n_permutations=20,
        seed=42,
        img_format="png",
        dpi=72,
    )
    return out / "data"


class TestPipelineSnapshot:
    @pytest.mark.parametrize("method", ["correlational", "discriminatory"])
    def test_deterministic_outputs(self, method, request):
        data = request.getfixturevalue(f"{method}_data")
        _snapshot(
            f"{method}_singular_values", _load(data / "singular_values.csv"), TIGHT
        )
        _snapshot(
            f"{method}_x_loadings", _load(data / "x_loadings.csv", index_col=0), TIGHT
        )
        _snapshot(
            f"{method}_y_loadings", _load(data / "y_loadings.csv", index_col=0), TIGHT
        )
        _snapshot(
            f"{method}_subject_scores",
            _load(data / "subject_scores.csv", index_col=0),
            TIGHT,
        )

    @pytest.mark.parametrize("method", ["correlational", "discriminatory"])
    def test_permutation_pvalues(self, method, request):
        data = request.getfixturevalue(f"{method}_data")
        _snapshot(
            f"{method}_p_values",
            _load(data / "p_values.csv"),
            PVAL,
            cols=_significant_lvs(method),
        )

    @pytest.mark.parametrize("method", ["correlational", "discriminatory"])
    def test_bootstrap_ratios(self, method, request):
        data = request.getfixturevalue(f"{method}_data")
        sig = _significant_lvs(method)
        _snapshot(
            f"{method}_x_bootstrap_ratios",
            _load(data / "x_bootstrap_ratios.csv", index_col=0),
            LOOSE,
            cols=sig,
        )
        _snapshot(
            f"{method}_y_bootstrap_ratios",
            _load(data / "y_bootstrap_ratios.csv", index_col=0),
            LOOSE,
            cols=sig,
        )

    def test_cross_validation(self, cv_data):
        fold = pd.read_csv(cv_data / "cv_fold_results.csv")
        null = pd.read_csv(cv_data / "cv_permutation_accuracies.csv")
        _snapshot("cv_mean_accuracy", np.array([fold["accuracy"].mean()]), SCALAR)
        _snapshot(
            "cv_mean_null_accuracy", np.array([null["null_accuracy"].mean()]), SCALAR
        )
