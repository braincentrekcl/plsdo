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
