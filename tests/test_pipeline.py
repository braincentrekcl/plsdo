"""Tests for pipeline helpers."""

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from plsdo.pipeline import _plot_verbose, run_pipeline
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

    def test_multi_index_subject_scores_csv(self, tmp_path):
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
