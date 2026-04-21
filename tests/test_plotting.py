import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pathlib import Path
from plsdo.plotting import figure_size, plot_heatmap, plot_permutation


class TestFigureSize:
    def test_small_data(self):
        w, h = figure_size(5, 3)
        assert 4 <= w <= 40
        assert 4 <= h <= 40

    def test_large_data_capped(self):
        w, h = figure_size(200, 100)
        assert w <= 40
        assert h <= 40

    def test_tiny_data_floored(self):
        w, h = figure_size(1, 1)
        assert w >= 4
        assert h >= 4


class TestPlotHeatmap:
    def test_saves_file(self, tmp_output):
        data = np.random.default_rng(0).standard_normal((5, 4))
        out = tmp_output / "heatmap.svg"
        plot_heatmap(
            data, v=1.0,
            xticklabels=["a", "b", "c", "d"],
            yticklabels=["r1", "r2", "r3", "r4", "r5"],
            out_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_annotations_suppressed_for_large_data(self, tmp_output):
        data = np.random.default_rng(0).standard_normal((35, 35))
        out = tmp_output / "big_heatmap.svg"
        fig, ax = plot_heatmap(
            data, v=1.0,
            xticklabels=[f"x{i}" for i in range(35)],
            yticklabels=[f"y{i}" for i in range(35)],
            out_path=out,
            return_fig=True,
        )
        # Check no annotation text objects
        texts = [c for c in ax.get_children()
                 if isinstance(c, matplotlib.text.Text)
                 and c.get_text() not in ("", " ")]
        # Only tick labels and title, no cell annotations
        annotation_count = sum(
            1 for t in texts
            if t.get_position()[0] > 0 and t.get_position()[1] > 0
            and t.get_text().replace("-", "").replace(".", "").isdigit()
        )
        assert annotation_count == 0
        plt.close(fig)


class TestPlotPermutation:
    def test_saves_file(self, tmp_output):
        s = np.array([2.5, 1.5, 0.8])
        perm_s = np.random.default_rng(0).standard_normal((3, 100)).clip(0)
        p_vals = np.array([0.01, 0.05, 0.50])
        out = tmp_output / "perm.svg"
        plot_permutation(s, perm_s, p_vals, out_path=out)
        assert out.exists()
