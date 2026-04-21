"""Stateless plot functions for PLS results."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
