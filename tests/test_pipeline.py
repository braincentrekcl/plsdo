"""Tests for pipeline helpers."""

import logging
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from plsdo import core as core_mod
from plsdo import pipeline as pipeline_mod
from plsdo.io import GroupConfig, GroupSpec
from plsdo.pipeline import (
    _plot_score_boxstrips,
    _plot_verbose,
    cross_validate_pipeline,
    run_pipeline,
)
from plsdo.plotting import VERBOSE_FEATURE_LIMIT

DATA_DIR = Path(__file__).parent / "data"


def _make_mock_model(n_x: int, n_y: int, n_components: int = 2):
    """Build a minimal mock PLS model with the attributes _plot_verbose needs."""
    rng = np.random.default_rng(0)
    return SimpleNamespace(
        s=np.ones(n_components),
        p_values=np.full(n_components, 0.01),
        u=rng.standard_normal((n_x, n_components)),
        vt=rng.standard_normal((n_components, n_y)),
        u_bootstrap_ratios=rng.standard_normal((n_x, n_components)),
        vt_bootstrap_ratios=rng.standard_normal((n_components, n_y)),
    )


class TestVerboseFeatureLimit:
    """Verbose plot guard: skip feature-heavy plots above the limit."""

    @pytest.fixture()
    def figures_dir(self, tmp_path):
        d = tmp_path / "figures"
        d.mkdir()
        return d

    def test_guard_fires_at_default_limit(self, figures_dir, caplog):
        """101 features exceeds the default 100 — only scree produced."""
        n_features = VERBOSE_FEATURE_LIMIT + 1
        model = _make_mock_model(n_x=n_features, n_y=n_features)

        with caplog.at_level(logging.WARNING, logger="plsdo"):
            _plot_verbose(
                model=model,
                method="correlational",
                X=np.zeros((10, n_features)),
                Y=np.zeros((10, n_features)),
                x_feature_names=[f"x{i}" for i in range(n_features)],
                x_display_names=[f"x{i}" for i in range(n_features)],
                y_feature_names=[f"y{i}" for i in range(n_features)],
                x_colours=None,
                y_colours=None,
                final_lv_indices=np.array([0]),
                final_lv_names=["LV1"],
                config=None,
                demo_aligned=None,
                figures_dir=figures_dir,
                ext="svg",
                dpi=72,
            )

        # Guard warning was logged
        assert any("Skipping verbose plots" in msg for msg in caplog.messages)

        # Only scree was produced — no heatmaps or distributions
        produced = sorted(p.name for p in figures_dir.iterdir())
        assert produced == ["scree.svg"]

    def test_explicit_override_bypasses_guard(self, figures_dir, caplog):
        """150 features with limit=200 — guard does NOT fire."""
        n_features = 150
        model = _make_mock_model(n_x=n_features, n_y=n_features)

        with caplog.at_level(logging.WARNING, logger="plsdo"):
            _plot_verbose(
                model=model,
                method="correlational",
                X=np.zeros((10, n_features)),
                Y=np.zeros((10, n_features)),
                x_feature_names=[f"x{i}" for i in range(n_features)],
                x_display_names=[f"x{i}" for i in range(n_features)],
                y_feature_names=[f"y{i}" for i in range(n_features)],
                x_colours=None,
                y_colours=None,
                final_lv_indices=np.array([0]),
                final_lv_names=["LV1"],
                config=None,
                demo_aligned=None,
                figures_dir=figures_dir,
                ext="svg",
                dpi=72,
                verbose_feature_limit=200,
            )

        # No guard warning
        assert not any("Skipping verbose plots" in msg for msg in caplog.messages)

        # More than just scree was produced
        produced = sorted(p.name for p in figures_dir.iterdir())
        assert len(produced) > 1
        assert "scree.svg" in produced

    def test_explicit_lower_limit_fires_guard(self, figures_dir, caplog):
        """60 features with limit=50 — guard fires."""
        n_features = 60
        model = _make_mock_model(n_x=n_features, n_y=n_features)

        with caplog.at_level(logging.WARNING, logger="plsdo"):
            _plot_verbose(
                model=model,
                method="correlational",
                X=np.zeros((10, n_features)),
                Y=np.zeros((10, n_features)),
                x_feature_names=[f"x{i}" for i in range(n_features)],
                x_display_names=[f"x{i}" for i in range(n_features)],
                y_feature_names=[f"y{i}" for i in range(n_features)],
                x_colours=None,
                y_colours=None,
                final_lv_indices=np.array([0]),
                final_lv_names=["LV1"],
                config=None,
                demo_aligned=None,
                figures_dir=figures_dir,
                ext="svg",
                dpi=72,
                verbose_feature_limit=50,
            )

        assert any("Skipping verbose plots" in msg for msg in caplog.messages)
        produced = sorted(p.name for p in figures_dir.iterdir())
        assert produced == ["scree.svg"]


class TestMultiIndexSubjectScores:
    """Integration: compound subject ID produces a two-level index in CSV."""

    def test_multi_index_subject_scores_csv(self, tmp_path, monkeypatch):
        # The synthetic multi-index data keeps no LV on its own, so force one
        # to survive: the scores CSV is only written when there is a final LV.
        _force_one_significant_lv(monkeypatch)
        out = tmp_path / "output"
        run_pipeline(
            method="discriminatory",
            y_path=DATA_DIR / "behaviour_multi.csv",
            demographics_path=DATA_DIR / "demographics_multi.csv",
            output_dir=out,
            groups_path=DATA_DIR / "groups_multi.yaml",
            n_perms=20,
            n_bootstraps=20,
            seed=42,
            img_format="png",
            dpi=72,
        )
        scores = pd.read_csv(out / "data" / "subject_scores.csv")
        # Compound key: subject_id and run_id are both columns in the CSV
        assert "subject_id" in scores.columns
        assert "run_id" in scores.columns

    def test_single_index_output_unchanged(self, tmp_path):
        out = tmp_path / "output"
        run_pipeline(
            method="discriminatory",
            y_path=DATA_DIR / "behaviour.csv",
            demographics_path=DATA_DIR / "demographics.csv",
            output_dir=out,
            group_col="group",
            subject_id="subject_id",
            n_perms=20,
            n_bootstraps=20,
            seed=42,
            img_format="png",
            dpi=72,
        )
        scores = pd.read_csv(out / "data" / "subject_scores.csv")
        # Single key: subject_id is the index column
        assert "subject_id" in scores.columns
        # run_id should NOT be present
        assert "run_id" not in scores.columns


DATA_CSVS = [
    "singular_values.csv",
    "p_values.csv",
    "x_loadings.csv",
    "y_loadings.csv",
    "x_bootstrap_ratios.csv",
    "y_bootstrap_ratios.csv",
    "subject_scores.csv",
]


def _run(method, out):
    """Run a full pipeline on the synthetic data for the given method."""
    kwargs = dict(
        method=method,
        y_path=DATA_DIR / "behaviour.csv",
        demographics_path=DATA_DIR / "demographics.csv",
        output_dir=out,
        group_col="group",
        subject_id="subject_id",
        n_perms=50,
        n_bootstraps=50,
        seed=42,
        img_format="png",
        dpi=72,
    )
    if method == "correlational":
        kwargs["x_path"] = DATA_DIR / "brain.csv"
    run_pipeline(**kwargs)


class TestRunPipelineOutputs:
    """End-to-end: a full run writes the expected CSVs, figures, and log."""

    @pytest.fixture(params=["correlational", "discriminatory"])
    def run_out(self, request, tmp_path):
        out = tmp_path / "output"
        _run(request.param, out)
        return request.param, out

    def test_all_data_csvs_written(self, run_out):
        _method, out = run_out
        data = out / "data"
        for name in DATA_CSVS:
            assert (data / name).exists(), f"missing {name}"

    def test_log_written_with_version_and_params(self, run_out):
        from plsdo import __version__

        method, out = run_out
        log = (out / "log.txt").read_text()
        assert "PLS analysis log" in log
        assert __version__ in log
        assert f"method: {method}" in log

    def test_core_figures_produced(self, run_out):
        _method, out = run_out
        figs = out / "figures"
        assert (figs / "cross_correlation.png").exists()
        assert (figs / "permutation_test.png").exists()

    def test_singular_values_and_pvalues_schema(self, run_out):
        _method, out = run_out
        sv = pd.read_csv(out / "data" / "singular_values.csv")
        pv = pd.read_csv(out / "data" / "p_values.csv")
        # One row of values; columns are LV labels and identical across both.
        assert len(sv) == 1 and len(pv) == 1
        assert all(re.fullmatch(r"LV\d+", c) for c in sv.columns)
        assert list(sv.columns) == list(pv.columns)
        # p-values are valid probabilities.
        assert ((pv.iloc[0] >= 0.0) & (pv.iloc[0] <= 1.0)).all()

    def test_loadings_and_bsr_dimensions(self, run_out):
        method, out = run_out
        x_load = pd.read_csv(out / "data" / "x_loadings.csv", index_col=0)
        y_load = pd.read_csv(out / "data" / "y_loadings.csv", index_col=0)
        # Rows index features; correlational X has 5 brain features,
        # discriminatory X has 3 group dummies; Y always has 4 behaviour features.
        assert len(y_load.index) == 4
        assert len(x_load.index) == (5 if method == "correlational" else 3)
        # Bootstrap-ratio matrices match the loading matrices.
        x_bsr = pd.read_csv(out / "data" / "x_bootstrap_ratios.csv", index_col=0)
        y_bsr = pd.read_csv(out / "data" / "y_bootstrap_ratios.csv", index_col=0)
        assert x_bsr.shape == x_load.shape
        assert y_bsr.shape == y_load.shape

    def test_subject_scores_one_row_per_subject_with_paired_lvs(self, run_out):
        _method, out = run_out
        scores = pd.read_csv(out / "data" / "subject_scores.csv")
        assert len(scores) == 12
        # filter_lvs feeds the score CSV: surviving LVs appear as paired X_/Y_ columns.
        x_cols = [c for c in scores.columns if c.startswith("X_")]
        y_cols = [c for c in scores.columns if c.startswith("Y_")]
        assert len(x_cols) == len(y_cols)


def _force_no_significant_lvs(monkeypatch):
    """Make filter_lvs drop every LV, simulating a null result."""
    original = core_mod.PLS.filter_lvs

    def zero_filter(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.final_lvs = np.zeros(len(self.s), dtype=bool)

    monkeypatch.setattr(core_mod.PLS, "filter_lvs", zero_filter)


def _force_one_significant_lv(monkeypatch):
    """Make filter_lvs keep exactly the first LV, simulating a real result."""
    original = core_mod.PLS.filter_lvs

    def one_filter(self, *args, **kwargs):
        original(self, *args, **kwargs)
        mask = np.zeros(len(self.s), dtype=bool)
        mask[0] = True
        self.final_lvs = mask

    monkeypatch.setattr(core_mod.PLS, "filter_lvs", one_filter)


def _warning_records(caplog):
    return [r for r in caplog.records if r.levelname == "WARNING"]


class TestNullResultWarning:
    """A null result (no significant + reliable LV) must be announced loudly,
    not just left as an empty list at INFO and an index-only scores CSV."""

    def test_warns_when_no_lvs_survive(self, tmp_path, monkeypatch, caplog):
        _force_no_significant_lvs(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="plsdo"):
            _run("discriminatory", tmp_path / "out")
        assert any(
            "no latent variable" in r.message.lower() for r in _warning_records(caplog)
        )

    def test_no_warning_when_lvs_survive(self, tmp_path, monkeypatch, caplog):
        # A normal run on the synthetic data keeps at least one LV.
        with caplog.at_level(logging.WARNING, logger="plsdo"):
            _run("discriminatory", tmp_path / "out")
        assert not any(
            "no latent variable" in r.message.lower() for r in _warning_records(caplog)
        )

    def test_no_scores_csv_when_no_lvs_survive(self, tmp_path, monkeypatch):
        _force_no_significant_lvs(monkeypatch)
        out = tmp_path / "out"
        _run("discriminatory", out)
        assert not (out / "data" / "subject_scores.csv").exists()

    def test_null_result_recorded_in_log(self, tmp_path, monkeypatch):
        """The null-result warning must also be persisted durably in log.txt,
        not only emitted to the console."""
        _force_no_significant_lvs(monkeypatch)
        out = tmp_path / "out"
        _run("discriminatory", out)
        log = (out / "log.txt").read_text()
        assert "no latent variable" in log.lower()

    def test_normal_run_log_omits_null_message(self, tmp_path):
        """A normal run keeps at least one LV, so log.txt must not contain the
        null-result message."""
        out = tmp_path / "out"
        _run("discriminatory", out)
        log = (out / "log.txt").read_text()
        assert "no latent variable" not in log.lower()


class TestCrossValidatePipelineOutputs:
    """End-to-end: cross_validate_pipeline writes its CSVs, figures, and log."""

    @pytest.fixture
    def cv_out(self, tmp_path):
        out = tmp_path / "cv"
        cross_validate_pipeline(
            y_path=DATA_DIR / "behaviour.csv",
            demographics_path=DATA_DIR / "demographics.csv",
            output_dir=out,
            group_col="group",
            subject_id="subject_id",
            n_folds=2,
            n_repeats=3,
            n_permutations=20,
            seed=42,
            img_format="png",
            dpi=72,
        )
        return out

    def test_data_csvs_written(self, cv_out):
        data = cv_out / "data"
        assert (data / "cv_fold_results.csv").exists()
        assert (data / "cv_permutation_accuracies.csv").exists()

    def test_figures_produced(self, cv_out):
        figs = cv_out / "figures"
        assert (figs / "cv_fold_accuracy.png").exists()
        assert (figs / "cv_permutation_test.png").exists()
        assert (figs / "cv_confusion_matrix.png").exists()

    def test_log_records_cv_run(self, cv_out):
        log = (cv_out / "log.txt").read_text()
        assert "cross-validate" in log
        assert "mean_accuracy" in log

    def test_permutation_accuracies_count(self, cv_out):
        null = pd.read_csv(cv_out / "data" / "cv_permutation_accuracies.csv")
        assert len(null) == 20


class TestFacetWiring:
    """The pipeline must translate facet roles into FacetGrid row/col axes.

    LV is always shown, so it occupies one axis and at most one demographic
    facet can take the other. (The two-facet case is rejected at parse time;
    see test_io.py::TestParseGroupsConfig::test_both_facet_roles_raises.)
    """

    def _capture_boxstrip_calls(self, config, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(
            pipeline_mod, "plot_scores_boxstrip", lambda **kw: calls.append(kw)
        )
        n = 6
        rng = np.random.default_rng(0)
        model = SimpleNamespace(
            final_lvs=np.array([True, True]),
            x_scores=rng.standard_normal((n, 2)),
            y_scores=rng.standard_normal((n, 2)),
        )
        demo = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B", "C", "C"],
                "sex": ["F", "M", "F", "M", "F", "M"],
                "site": ["P", "Q", "P", "Q", "P", "Q"],
            }
        )
        _plot_score_boxstrips(
            model,
            config,
            demo,
            [f"s{i}" for i in range(n)],
            ["subject_id"],
            ["LV1", "LV2"],
            tmp_path,
            "svg",
            72,
        )
        return calls

    def test_default_keeps_lv_on_columns(self, monkeypatch, tmp_path):
        config = GroupConfig(groups=[GroupSpec("group", "x_axis")])
        calls = self._capture_boxstrip_calls(config, monkeypatch, tmp_path)
        assert calls[0]["col_col"] == "LV"
        assert calls[0]["row_col"] is None

    def test_facet_rows_adds_row_axis(self, monkeypatch, tmp_path):
        config = GroupConfig(
            groups=[GroupSpec("group", "x_axis"), GroupSpec("sex", "facet_rows")]
        )
        calls = self._capture_boxstrip_calls(config, monkeypatch, tmp_path)
        assert calls[0]["col_col"] == "LV"
        assert calls[0]["row_col"] == "sex"

    def test_facet_cols_moves_lv_to_rows(self, monkeypatch, tmp_path):
        config = GroupConfig(
            groups=[GroupSpec("group", "x_axis"), GroupSpec("sex", "facet_cols")]
        )
        calls = self._capture_boxstrip_calls(config, monkeypatch, tmp_path)
        assert calls[0]["col_col"] == "sex"
        assert calls[0]["row_col"] == "LV"

    def test_default_layout_threads_col_wrap(self, monkeypatch, tmp_path):
        """In the default layout (LV on columns, no facet) facet_col_wrap is
        passed through to control column wrapping."""
        config = GroupConfig(
            groups=[GroupSpec("group", "x_axis", facet_col_wrap=3)]
        )
        calls = self._capture_boxstrip_calls(config, monkeypatch, tmp_path)
        assert calls[0]["col_col"] == "LV"
        assert calls[0]["row_col"] is None
        assert calls[0]["col_wrap"] == 3

    def test_facet_col_wrap_inert_with_facet_role_warns(
        self, monkeypatch, tmp_path, caplog
    ):
        """facet_col_wrap cannot apply alongside a facet role (the columns
        can't wrap when LV shares an axis with the facet); warn rather than
        discard it silently."""
        config = GroupConfig(
            groups=[
                GroupSpec("group", "x_axis"),
                GroupSpec("sex", "facet_rows", facet_col_wrap=3),
            ]
        )
        with caplog.at_level("WARNING", logger="plsdo"):
            calls = self._capture_boxstrip_calls(config, monkeypatch, tmp_path)
        assert calls[0]["col_wrap"] is None
        assert "facet_col_wrap" in caplog.text
