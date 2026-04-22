import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pathlib import Path
from plsdo.plotting import (
    figure_size, plot_heatmap, plot_permutation,
    plot_loadings, plot_scores_boxstrip, plot_scores_scatter,
    plot_cv_accuracy, plot_cv_permutation, plot_confusion_matrix,
    plot_lv_heatmap, plot_bootstrap_heatmap, plot_raw_distributions,
    plot_scree, plot_cv_convergence,
)


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


class TestPlotLoadings:
    def test_saves_file(self, tmp_output):
        loadings = np.array([0.8, -0.5, 0.3, -0.9, 0.1])
        se = np.array([0.1, 0.2, 0.15, 0.1, 0.3])
        out = tmp_output / "loadings.svg"
        plot_loadings(
            loadings=loadings,
            se=se,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            lv_name="LV1",
            out_path=out,
        )
        assert out.exists()

    def test_with_category_colours(self, tmp_output):
        loadings = np.array([0.8, -0.5, 0.3])
        se = np.array([0.1, 0.2, 0.15])
        colours = ["red", "red", "blue"]
        out = tmp_output / "loadings_coloured.svg"
        plot_loadings(
            loadings=loadings,
            se=se,
            feature_names=["f1", "f2", "f3"],
            lv_name="LV1",
            out_path=out,
            colours=colours,
        )
        assert out.exists()


class TestPlotScoresBoxstrip:
    def test_saves_file(self, tmp_output):
        import pandas as pd

        scores_df = pd.DataFrame({
            "score": np.random.default_rng(0).standard_normal(12),
            "LV": ["LV1"] * 6 + ["LV2"] * 6,
            "group": ["A", "A", "B", "B", "C", "C"] * 2,
        })
        scores_df["group"] = pd.Categorical(
            scores_df["group"], categories=["A", "B", "C"], ordered=True
        )
        out = tmp_output / "scores_box.svg"
        plot_scores_boxstrip(
            scores_df=scores_df,
            x_col="group",
            y_col="score",
            col_col="LV",
            out_path=out,
        )
        assert out.exists()


class TestPlotScoresScatter:
    def test_saves_file(self, tmp_output):
        import pandas as pd

        scatter_df = pd.DataFrame({
            "x_score": np.random.default_rng(0).standard_normal(12),
            "y_score": np.random.default_rng(1).standard_normal(12),
            "group": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
        })
        out = tmp_output / "scatter.svg"
        plot_scores_scatter(
            scatter_df=scatter_df,
            x_col="x_score",
            y_col="y_score",
            hue_col="group",
            lv_name="LV1",
            out_path=out,
        )
        assert out.exists()


class TestCVPlots:
    def test_cv_accuracy_saves(self, tmp_output):
        accs = np.random.default_rng(0).uniform(0.2, 0.8, size=50)
        out = tmp_output / "cv_acc.svg"
        plot_cv_accuracy(
            fold_accuracies=accs, mean_accuracy=0.5,
            chance_level=0.25, out_path=out,
        )
        assert out.exists()

    def test_cv_permutation_saves(self, tmp_output):
        null_accs = np.random.default_rng(0).uniform(0.2, 0.4, size=100)
        out = tmp_output / "cv_perm.svg"
        plot_cv_permutation(
            null_accuracies=null_accs, observed_accuracy=0.6,
            p_value=0.01, out_path=out,
        )
        assert out.exists()

    def test_confusion_matrix_saves(self, tmp_output):
        cm = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
        out = tmp_output / "cm.svg"
        plot_confusion_matrix(
            cm=cm, label_names=["A", "B", "C"],
            mean_accuracy=0.73, out_path=out,
        )
        assert out.exists()


class TestPlotLvHeatmap:
    def test_saves_file(self, tmp_output):
        rng = np.random.default_rng(0)
        u = rng.standard_normal((5, 3))
        s = np.array([2.0, 1.5, 0.5])
        vt = rng.standard_normal((3, 4))
        out = tmp_output / "lv_heatmap.svg"
        plot_lv_heatmap(
            lv_idx=0, u=u, s=s, vt=vt,
            x_feature_names=["x1", "x2", "x3", "x4", "x5"],
            y_feature_names=["y1", "y2", "y3", "y4"],
            out_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_each_lv_different_output(self, tmp_output):
        rng = np.random.default_rng(1)
        u = rng.standard_normal((5, 3))
        s = np.array([2.0, 1.5, 0.5])
        vt = rng.standard_normal((3, 4))
        x_names = ["x1", "x2", "x3", "x4", "x5"]
        y_names = ["y1", "y2", "y3", "y4"]
        for i in range(3):
            out = tmp_output / f"lv{i}_heatmap.svg"
            plot_lv_heatmap(lv_idx=i, u=u, s=s, vt=vt,
                            x_feature_names=x_names, y_feature_names=y_names,
                            out_path=out)
            assert out.exists()


class TestPlotBootstrapHeatmap:
    def test_saves_file(self, tmp_output):
        rng = np.random.default_rng(0)
        bsr = rng.standard_normal((5, 2))
        out = tmp_output / "bsr_heatmap.svg"
        plot_bootstrap_heatmap(
            bootstrap_ratios=bsr,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            lv_names=["LV1", "LV2"],
            out_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_single_lv(self, tmp_output):
        rng = np.random.default_rng(0)
        bsr = rng.standard_normal((4, 1))
        out = tmp_output / "bsr_single.svg"
        plot_bootstrap_heatmap(
            bootstrap_ratios=bsr,
            feature_names=["f1", "f2", "f3", "f4"],
            lv_names=["LV1"],
            out_path=out,
        )
        assert out.exists()


class TestPlotRawDistributions:
    def test_saves_file(self, tmp_output):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((12, 3))
        group_labels = ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"]
        out = tmp_output / "raw_dist.svg"
        plot_raw_distributions(
            data=data,
            feature_names=["feat1", "feat2", "feat3"],
            group_labels=group_labels,
            group_col="group",
            out_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_single_feature(self, tmp_output):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((8, 1))
        group_labels = ["A"] * 4 + ["B"] * 4
        out = tmp_output / "raw_dist_single.svg"
        plot_raw_distributions(
            data=data,
            feature_names=["feat1"],
            group_labels=group_labels,
            group_col="group",
            out_path=out,
        )
        assert out.exists()


class TestPlotScree:
    def test_saves_file(self, tmp_output):
        s = np.array([2.5, 1.5, 0.8, 0.4])
        p_values = np.array([0.01, 0.03, 0.20, 0.80])
        out = tmp_output / "scree.svg"
        plot_scree(s=s, p_values=p_values, out_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_all_significant(self, tmp_output):
        s = np.array([3.0, 2.0, 1.0])
        p_values = np.array([0.001, 0.01, 0.04])
        out = tmp_output / "scree_sig.svg"
        plot_scree(s=s, p_values=p_values, out_path=out)
        assert out.exists()

    def test_none_significant(self, tmp_output):
        s = np.array([1.0, 0.8])
        p_values = np.array([0.10, 0.50])
        out = tmp_output / "scree_ns.svg"
        plot_scree(s=s, p_values=p_values, out_path=out)
        assert out.exists()


class TestPlotCvConvergence:
    def test_saves_file(self, tmp_output):
        rng = np.random.default_rng(0)
        repeat_accs = rng.uniform(0.4, 0.7, size=20)
        out = tmp_output / "cv_convergence.svg"
        plot_cv_convergence(
            repeat_accuracies=repeat_accs,
            final_mean=float(repeat_accs.mean()),
            chance_level=0.25,
            out_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_single_repeat(self, tmp_output):
        out = tmp_output / "cv_conv_single.svg"
        plot_cv_convergence(
            repeat_accuracies=np.array([0.60]),
            final_mean=0.60,
            chance_level=0.33,
            out_path=out,
        )
        assert out.exists()
