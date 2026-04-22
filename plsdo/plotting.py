"""Stateless plot functions for PLS results."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Optional

# Threshold above which heatmap annotations are suppressed
ANNOTATION_THRESHOLD = 30


def figure_size(
    n_rows: int, n_cols: int,
    cell_size: float = 0.5,
    min_dim: float = 4.0,
    max_dim: float = 40.0,
) -> tuple[float, float]:
    """Compute figure dimensions from data shape.

    Parameters
    ----------
    n_rows, n_cols : int
        Data dimensions.
    cell_size : float
        Size per cell in inches.
    min_dim, max_dim : float
        Floor and ceiling for each dimension.

    Returns
    -------
    (width, height) in inches.
    """
    width = max(min_dim, min(max_dim, n_cols * cell_size + 2))
    height = max(min_dim, min(max_dim, n_rows * cell_size + 2))
    return (width, height)


def plot_heatmap(
    data: np.ndarray,
    v: float,
    xticklabels: list[str],
    yticklabels: list[str],
    out_path: Path,
    subtitle: Optional[str] = None,
    row_colors: Optional[list] = None,
    col_colors: Optional[list] = None,
    dpi: int = 300,
    return_fig: bool = False,
) -> Optional[tuple]:
    """Plot a heatmap with diverging colour scale.

    Reference: correlational_pls.ipynb heatmapplot function.

    Parameters
    ----------
    data : ndarray, shape (n_rows, n_cols)
    v : float
        Symmetric colour range [-v, v].
    xticklabels, yticklabels : list of str
    out_path : Path
    subtitle : str, optional
    row_colors, col_colors : list, optional
        Colour bars for row/column groupings.
    dpi : int
    return_fig : bool
        If True, return (fig, ax) instead of closing.
    """
    n_rows, n_cols = data.shape
    figsize = figure_size(n_rows, n_cols)
    annotate = n_rows <= ANNOTATION_THRESHOLD and n_cols <= ANNOTATION_THRESHOLD

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        data,
        vmin=-v, vmax=v, center=0,
        cmap="vlag",
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        ax=ax,
        cbar_kws={"shrink": 0.5},
        annot=annotate,
        fmt=".2f" if annotate else "",
    )
    if subtitle:
        fig.suptitle(subtitle)
    plt.tight_layout()
    fig.savefig(out_path, transparent=False, dpi=dpi)

    if return_fig:
        return fig, ax
    plt.close(fig)
    return None


def plot_permutation(
    observed_s: np.ndarray,
    permuted_s: np.ndarray,
    p_values: np.ndarray,
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Plot permutation test histograms.

    Reference: correlational_pls.ipynb cell 38.

    Parameters
    ----------
    observed_s : ndarray, shape (n_components,)
    permuted_s : ndarray, shape (n_components, n_perms)
    p_values : ndarray, shape (n_components,)
    out_path : Path
    """
    n_components = len(observed_s)
    n_cols = min(4, n_components)
    n_rows = int(np.ceil(n_components / n_cols))

    fig, axs = plt.subplots(
        n_rows, n_cols, sharex=True, sharey=True,
        figsize=(3 * n_cols, 2.5 * n_rows),
    )
    axs_flat = np.atleast_1d(axs).flat

    for idx, ax in enumerate(axs_flat):
        if idx < n_components:
            ax.hist(permuted_s[idx, :], color="grey", bins=100)
            ax.axvline(observed_s[idx], color="red", linestyle="--")
            ax.set_title(f"LV{idx + 1} (p={p_values[idx]:.4f})")
            ax.set_xlabel("Singular value")
            ax.set_ylabel("Count")
        else:
            ax.remove()

    fig.legend(["Observed"], loc="upper right")
    fig.suptitle("Singular value vs null distribution", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, transparent=False, dpi=dpi)
    plt.close(fig)


def plot_loadings(
    loadings: np.ndarray,
    se: np.ndarray,
    feature_names: list[str],
    lv_name: str,
    out_path: Path,
    colours: Optional[list] = None,
    dpi: int = 300,
) -> None:
    """Plot horizontal loading bar chart with SE error bars.

    Reference: correlational_pls.ipynb cells 48-49.

    Parameters
    ----------
    loadings : ndarray, shape (n_features,)
    se : ndarray, shape (n_features,)
    feature_names : list of str
    lv_name : str
        E.g. "LV1".
    out_path : Path
    colours : list, optional
        Per-feature colours from metadata categories.
    """
    n_features = len(loadings)
    sort_idx = np.argsort(np.abs(loadings))

    sorted_loadings = loadings[sort_idx]
    sorted_se = se[sort_idx]
    sorted_names = [feature_names[i] for i in sort_idx]
    sorted_colours = (
        [colours[i] for i in sort_idx] if colours is not None
        else ["steelblue"] * n_features
    )

    height = max(4, n_features * 0.3 + 1)
    fig, ax = plt.subplots(figsize=(8, height))
    ax.barh(
        np.arange(n_features),
        sorted_loadings,
        xerr=sorted_se,
        tick_label=sorted_names,
        color=sorted_colours,
        ecolor="red",
    )
    ax.set_title(f"Loadings for {lv_name}")
    plt.tight_layout()
    fig.savefig(out_path, transparent=False, dpi=dpi)
    plt.close(fig)


def plot_scores_boxstrip(
    scores_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    col_col: str,
    out_path: Path,
    hue_col: Optional[str] = None,
    row_col: Optional[str] = None,
    col_wrap: Optional[int] = None,
    dpi: int = 300,
) -> None:
    """Plot box/strip plots of subject scores by group.

    Reference: correlational_pls.ipynb boxstripplot function.

    Parameters
    ----------
    scores_df : DataFrame
        Long-format dataframe with score, LV, and group columns.
    x_col : str
        Column for x-axis categories.
    y_col : str
        Column for score values.
    col_col : str
        Column for facet columns.
    out_path : Path
    hue_col : str, optional
        Column for colour encoding.
    row_col : str, optional
        Column for facet rows.
    col_wrap : int, optional
        Number of facet columns before wrapping.
    """
    hue = hue_col or x_col
    order = (
        scores_df[x_col].cat.categories.tolist()
        if hasattr(scores_df[x_col], "cat")
        else sorted(scores_df[x_col].unique())
    )

    kwargs = {}
    if col_wrap is not None:
        kwargs["col_wrap"] = col_wrap
    elif row_col is not None:
        kwargs["row"] = row_col
    else:
        kwargs["col_wrap"] = 2

    g = sns.catplot(
        data=scores_df,
        x=x_col, y=y_col, hue=hue, col=col_col,
        order=order,
        kind="box",
        sharex=False,
        palette="Set2",
        boxprops={"edgecolor": "gray", "alpha": 0.5},
        medianprops={"color": "k", "ls": "--", "lw": 1},
        whiskerprops={"color": "gray", "ls": "-", "lw": 1},
        showfliers=False,
        legend_out=True,
        **kwargs,
    )
    g.map(
        sns.stripplot, x_col, y_col, hue,
        order=order,
        size=5, dodge=True, palette="Set2",
        jitter=True, linewidth=1, edgecolor=".5",
    )
    plt.tight_layout()
    g.savefig(out_path, transparent=False, dpi=dpi)
    plt.close()


def plot_scores_scatter(
    scatter_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    lv_name: str,
    out_path: Path,
    col_col: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """Plot XU vs YV' score scatter with per-group linear fits.

    Reference: correlational_pls.ipynb cells 75-78.

    Parameters
    ----------
    scatter_df : DataFrame
    x_col, y_col : str
        Columns for X and Y scores.
    hue_col : str
        Column for group colouring.
    lv_name : str
    out_path : Path
    col_col : str, optional
        Column for facetting.
    """
    kwargs = {"col": col_col} if col_col else {}
    g = sns.lmplot(
        data=scatter_df,
        x=x_col, y=y_col, hue=hue_col,
        palette="Set2",
        scatter_kws={"edgecolor": "gray"},
        line_kws={"alpha": 0.5, "linestyle": "--"},
        ci=95,
        **kwargs,
    )
    g.set_axis_labels(x_var="X score", y_var="Y score")
    g.figure.suptitle(f"Score scatter for {lv_name}", y=1.02)
    plt.tight_layout()
    g.savefig(out_path, transparent=False, dpi=dpi)
    plt.close()


def plot_cv_accuracy(
    fold_accuracies: np.ndarray,
    mean_accuracy: float,
    chance_level: float,
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Plot histogram of per-fold CV accuracies.

    Reference: claude_cross_validation.py section 7a.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(fold_accuracies, bins=20, color="steelblue", edgecolor="white")
    ax.axvline(
        mean_accuracy, color="red", linestyle="--",
        label=f"Mean = {mean_accuracy:.3f}",
    )
    ax.axvline(
        chance_level, color="gray", linestyle=":",
        label=f"Chance = {chance_level:.3f}",
    )
    ax.set_xlabel("Fold accuracy")
    ax.set_ylabel("Count")
    ax.set_title("Per-fold accuracy distribution")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, transparent=False, dpi=dpi)
    plt.close(fig)


def plot_cv_permutation(
    null_accuracies: np.ndarray,
    observed_accuracy: float,
    p_value: float,
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Plot null distribution of permuted CV accuracies.

    Reference: claude_cross_validation.py section 7b.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        null_accuracies, bins=40, color="gray",
        edgecolor="white", alpha=0.7,
    )
    ax.axvline(
        observed_accuracy, color="red", linestyle="--", linewidth=2,
        label=f"Observed = {observed_accuracy:.3f}",
    )
    ax.set_xlabel("Mean CV accuracy (permuted labels)")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation test (p = {p_value:.4f})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, transparent=False, dpi=dpi)
    plt.close(fig)


def meta_colours(
    meta_df,
    feature_names: list[str],
) -> Optional[list]:
    """Build a per-feature colour list from a metadata DataFrame.

    Parameters
    ----------
    meta_df : DataFrame or None
        Metadata with 'feature' and 'category' columns.
    feature_names : list of str
        Ordered feature names from the data matrix.

    Returns
    -------
    list of colours, or None if meta_df is None or has no 'category' column.
    """
    if meta_df is None or "category" not in meta_df.columns:
        return None
    colour_map = dict(zip(meta_df["feature"], meta_df["category"]))
    palette = dict(zip(
        meta_df["category"].unique(),
        sns.color_palette("Set2", n_colors=meta_df["category"].nunique()),
    ))
    return [
        palette.get(colour_map.get(f), "steelblue")
        for f in feature_names
    ]


def plot_lv_heatmap(
    lv_idx: int,
    u: np.ndarray,
    s: np.ndarray,
    vt: np.ndarray,
    x_feature_names: list[str],
    y_feature_names: list[str],
    out_path: Path,
    x_colours: Optional[list] = None,
    y_colours: Optional[list] = None,
    dpi: int = 300,
) -> None:
    """Plot the rank-1 reconstruction of R for a single latent variable.

    Reference: correlational_pls.ipynb cells 33-34.

    Parameters
    ----------
    lv_idx : int
        Zero-based index of the latent variable to plot.
    u : ndarray, shape (n_x, n_components)
        Left singular vectors from SVD.
    s : ndarray, shape (n_components,)
        Singular values.
    vt : ndarray, shape (n_components, n_y)
        Right singular vectors (transposed) from SVD.
    x_feature_names : list of str
    y_feature_names : list of str
    out_path : Path
    x_colours, y_colours : list, optional
        Row/column colour bars from metadata categories.
    dpi : int
    """
    lv_matrix = s[lv_idx] * np.outer(u[:, lv_idx], vt[lv_idx, :])
    plot_heatmap(
        lv_matrix, v=1.0,
        xticklabels=y_feature_names,
        yticklabels=x_feature_names,
        out_path=out_path,
        subtitle=f"LV{lv_idx + 1} rank-1 reconstruction",
        row_colors=x_colours,
        col_colors=y_colours,
        dpi=dpi,
    )


def plot_bootstrap_heatmap(
    bootstrap_ratios: np.ndarray,
    feature_names: list[str],
    lv_names: list[str],
    out_path: Path,
    v: float = 3.5,
    colours: Optional[list] = None,
    dpi: int = 300,
) -> None:
    """Plot a heatmap of bootstrap ratios for one side (X or Y).

    Reference: correlational_pls.ipynb cells 45-46.

    Parameters
    ----------
    bootstrap_ratios : ndarray, shape (n_features, n_sig_lvs)
        Bootstrap ratio matrix — rows are features, columns are LVs.
    feature_names : list of str
        Row labels (feature names).
    lv_names : list of str
        Column labels (e.g. ["LV1", "LV2"]).
    out_path : Path
    v : float
        Symmetric colour range [-v, v]. Default 3.5 (≈ 3.5 SE threshold).
    colours : list, optional
        Row colour bar from metadata categories.
    dpi : int
    """
    plot_heatmap(
        bootstrap_ratios, v=v,
        xticklabels=lv_names,
        yticklabels=feature_names,
        out_path=out_path,
        row_colors=colours,
        dpi=dpi,
    )


def plot_raw_distributions(
    data: np.ndarray,
    feature_names: list[str],
    group_labels,
    group_col: str,
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Plot z-scored feature distributions by group (box + strip).

    Reference: correlational_pls.ipynb cells 15-20 (boxstripplot on
    standardised X and Y before PLS).

    Parameters
    ----------
    data : ndarray, shape (n_subjects, n_features)
        Z-scored feature matrix.
    feature_names : list of str
        Column labels for *data*.
    group_labels : array-like, length n_subjects
        Per-subject group label for the x-axis.
    group_col : str
        Name to use for the group column (axis label).
    out_path : Path
    dpi : int
    """
    df = pd.DataFrame(data, columns=feature_names)
    df[group_col] = list(group_labels)

    long_df = df.melt(
        id_vars=[group_col],
        value_vars=feature_names,
        var_name="feature",
        value_name="z-score",
    )

    order = sorted(long_df[group_col].unique())
    n_features = len(feature_names)
    col_wrap = min(4, n_features)

    g = sns.catplot(
        data=long_df,
        x=group_col, y="z-score",
        hue=group_col,
        col="feature", col_wrap=col_wrap,
        order=order,
        kind="box",
        palette="Set2",
        boxprops={"edgecolor": "gray", "alpha": 0.5},
        medianprops={"color": "k", "ls": "--", "lw": 1},
        whiskerprops={"color": "gray", "ls": "-", "lw": 1},
        showfliers=False,
        legend_out=True,
        sharex=False,
    )
    g.map(
        sns.stripplot, group_col, "z-score", group_col,
        order=order,
        size=5, dodge=True, palette="Set2",
        jitter=True, linewidth=1, edgecolor=".5",
    )
    plt.tight_layout()
    g.savefig(out_path, transparent=False, dpi=dpi)
    plt.close()


def plot_scree(
    s: np.ndarray,
    p_values: np.ndarray,
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Plot singular value scree chart coloured by significance.

    Bars are coloured steelblue for significant LVs (p < 0.05) and
    lightgray for non-significant ones.

    Parameters
    ----------
    s : ndarray, shape (n_components,)
        Singular values.
    p_values : ndarray, shape (n_components,)
        Permutation p-values for each LV.
    out_path : Path
    dpi : int
    """
    from matplotlib.patches import Patch

    n = len(s)
    colours = ["steelblue" if p < 0.05 else "lightgray" for p in p_values]
    lv_labels = [f"LV{i + 1}" for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(4.0, n * 0.6 + 1.0), 4))
    ax.bar(lv_labels, s, color=colours, edgecolor="gray")
    ax.set_xlabel("Latent variable")
    ax.set_ylabel("Singular value")
    ax.set_title("Singular value scree plot")

    legend_elements = [
        Patch(facecolor="steelblue", label="p < 0.05"),
        Patch(facecolor="lightgray", label="p \u2265 0.05"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path, transparent=False, dpi=dpi)
    plt.close(fig)


def plot_cv_convergence(
    repeat_accuracies: np.ndarray,
    final_mean: float,
    chance_level: float,
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Plot cumulative mean CV accuracy over repeats (convergence check).

    Reference: claude_cross_validation.py section 7e.

    Parameters
    ----------
    repeat_accuracies : ndarray, shape (n_repeats,)
        Per-repeat mean accuracy (one value per repeat).
    final_mean : float
        Overall mean accuracy across all repeats.
    chance_level : float
        Chance accuracy (1 / n_groups).
    out_path : Path
    dpi : int
    """
    n = len(repeat_accuracies)
    cumulative_mean = np.cumsum(repeat_accuracies) / np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, n + 1), cumulative_mean, color="steelblue")
    ax.axhline(
        final_mean, color="red", linestyle="--", alpha=0.5,
        label=f"Final mean = {final_mean:.3f}",
    )
    ax.axhline(
        chance_level, color="gray", linestyle=":",
        label=f"Chance = {chance_level:.3f}",
    )
    ax.set_xlabel("Number of repeats completed")
    ax.set_ylabel("Cumulative mean accuracy")
    ax.set_title("Convergence of CV accuracy estimate with increasing repeats")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, transparent=False, dpi=dpi)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: list[str],
    mean_accuracy: float,
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Plot normalised confusion matrix heatmap.

    Reference: claude_cross_validation.py section 7c.
    """
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=label_names,
    )
    disp.plot(ax=ax, cmap="Blues", values_format=".0%")
    ax.set_title(f"CV confusion matrix\nAccuracy: {mean_accuracy:.1%}")
    plt.tight_layout()
    fig.savefig(out_path, transparent=False, dpi=dpi)
    plt.close(fig)
