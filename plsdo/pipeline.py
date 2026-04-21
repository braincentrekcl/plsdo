"""Pipeline orchestration for PLS analysis."""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from plsdo import __version__
from plsdo.core import PLS
from plsdo.io import (
    GroupConfig,
    align_subjects,
    build_design_matrix,
    check_missing_values,
    check_variance,
    detect_subject_id,
    load_csv,
    load_metadata,
    parse_groups_config,
    zscore_columns,
)
from plsdo.plotting import (
    plot_heatmap,
    plot_loadings,
    plot_permutation,
    plot_scores_boxstrip,
    plot_scores_scatter,
)

logger = logging.getLogger("plsdo")


def _write_log(output_dir: Path, params: dict) -> None:
    """Write a log.txt with run parameters."""
    log_path = output_dir / "log.txt"
    with open(log_path, "w") as f:
        f.write("PLS analysis log\n")
        f.write(f"Version: {__version__}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("\nParameters:\n")
        for k, v in params.items():
            f.write(f"  {k}: {v}\n")


def _save_csv(data: np.ndarray, path: Path, columns=None, index=None):
    """Save a numpy array as CSV."""
    df = pd.DataFrame(data, columns=columns, index=index)
    df.to_csv(path, index=index is not None)


def run_pipeline(
    *,
    method: str,
    y_path: Path,
    demographics_path: Path,
    output_dir: Path,
    x_path: Path | None = None,
    group_col: str | None = None,
    groups_path: Path | None = None,
    subject_id: str | None = None,
    x_meta_path: Path | None = None,
    y_meta_path: Path | None = None,
    n_perms: int = 10000,
    n_bootstraps: int = 10000,
    seed: int = 42,
    img_format: str = "svg",
    dpi: int = 300,
) -> None:
    """Run a full PLS analysis pipeline.

    Parameters
    ----------
    method : str
        "correlational" or "discriminatory".
    y_path : Path
        Path to Y matrix CSV.
    demographics_path : Path
        Path to demographics CSV.
    output_dir : Path
        Output directory for results.
    x_path : Path, optional
        Path to X matrix CSV (required for correlational).
    group_col : str, optional
        Single grouping column name.
    groups_path : Path, optional
        Path to YAML groups config.
    subject_id : str, optional
        Subject ID column name.
    x_meta_path, y_meta_path : Path, optional
        Paths to feature metadata CSVs.
    n_perms : int
        Number of permutations.
    n_bootstraps : int
        Number of bootstrap resamples.
    seed : int
        Random seed.
    img_format : str
        "svg" or "png".
    dpi : int
        Output image resolution.
    """
    # --- Set up output directory ---
    figures_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- Parse groups config ---
    demographics_df = load_csv(demographics_path)

    if groups_path is not None:
        config = parse_groups_config(groups_path, demographics_df=demographics_df)
        if config.subject_id:
            subject_id = config.subject_id
    elif group_col is not None:
        config = GroupConfig.from_group_col(group_col, subject_id=subject_id)
    else:
        config = None

    # --- Load and validate data ---
    y_df = load_csv(y_path, require_numeric=True)
    dfs_to_align = [y_df, demographics_df]

    if method == "correlational":
        x_df = load_csv(x_path, require_numeric=True)
        dfs_to_align.insert(0, x_df)

    sid = detect_subject_id(dfs_to_align, subject_id=subject_id)
    aligned = align_subjects(dfs_to_align, subject_id=sid)

    if method == "correlational":
        x_aligned, y_aligned, demo_aligned = aligned
        x_feature_names = [c for c in x_aligned.columns if c != sid]
    else:
        y_aligned, demo_aligned = aligned
        X_design, x_feature_names = build_design_matrix(demo_aligned, config)

    y_feature_names = [c for c in y_aligned.columns if c != sid]

    # --- Missing value checks ---
    check_missing_values(y_aligned, name="Y")
    if method == "correlational":
        check_missing_values(x_aligned, name="X")

    # --- Extract numeric arrays ---
    Y_raw = y_aligned[y_feature_names].to_numpy(dtype=float)

    if method == "correlational":
        X_raw = x_aligned[x_feature_names].to_numpy(dtype=float)
        check_variance(X_raw, feature_names=x_feature_names)
        check_variance(Y_raw, feature_names=y_feature_names)
        X = zscore_columns(X_raw)
        Y = zscore_columns(Y_raw)
    else:
        check_variance(Y_raw, feature_names=y_feature_names)
        X = X_design
        Y = zscore_columns(Y_raw)

    # --- Load metadata if provided ---
    x_meta_df = load_metadata(x_meta_path, x_feature_names) if x_meta_path else None
    y_meta_df = load_metadata(y_meta_path, y_feature_names) if y_meta_path else None

    # --- Run PLS ---
    model = PLS(X, Y, seed=seed)
    model.fit()
    model.permutation_test(n_perms=n_perms)
    model.bootstrap(n_bootstraps=n_bootstraps)
    model.filter_lvs()

    # --- Save data CSVs ---
    _save_csv(model.s[None, :], data_dir / "singular_values.csv",
              columns=[f"LV{i+1}" for i in range(len(model.s))])
    _save_csv(model.p_values[None, :], data_dir / "p_values.csv",
              columns=[f"LV{i+1}" for i in range(len(model.p_values))])
    _save_csv(model.u_loadings, data_dir / "x_loadings.csv",
              columns=[f"LV{i+1}" for i in range(model.u_loadings.shape[1])],
              index=x_feature_names)
    _save_csv(model.vt_loadings.T, data_dir / "y_loadings.csv",
              columns=[f"LV{i+1}" for i in range(model.vt_loadings.shape[0])],
              index=y_feature_names)
    _save_csv(model.u_bootstrap_ratios, data_dir / "x_bootstrap_ratios.csv",
              columns=[f"LV{i+1}" for i in range(model.u_bootstrap_ratios.shape[1])],
              index=x_feature_names)
    _save_csv(model.vt_bootstrap_ratios.T, data_dir / "y_bootstrap_ratios.csv",
              columns=[f"LV{i+1}" for i in range(model.vt_bootstrap_ratios.shape[0])],
              index=y_feature_names)

    # Subject scores
    subject_ids = y_aligned[sid].tolist()
    final_lv_names = [f"LV{i+1}" for i, v in enumerate(model.final_lvs) if v]
    scores_data = np.column_stack([
        model.x_scores[:, model.final_lvs],
        model.y_scores[:, model.final_lvs],
    ]) if any(model.final_lvs) else np.empty((len(subject_ids), 0))
    scores_cols = (
        [f"X_{name}" for name in final_lv_names]
        + [f"Y_{name}" for name in final_lv_names]
    )
    scores_df = pd.DataFrame(scores_data, columns=scores_cols, index=subject_ids)
    scores_df.index.name = sid
    scores_df.to_csv(data_dir / "subject_scores.csv")

    # --- Generate plots ---
    ext = img_format

    # 1. Cross-correlation heatmap
    plot_heatmap(
        model.xcorr, v=1.0,
        xticklabels=y_feature_names,
        yticklabels=x_feature_names,
        out_path=figures_dir / f"cross_correlation.{ext}",
        dpi=dpi,
    )

    # 2. Permutation test
    plot_permutation(
        model.s, model.permuted_singular_values, model.p_values,
        out_path=figures_dir / f"permutation_test.{ext}",
        dpi=dpi,
    )

    # 3. Loading bar plots for final LVs
    for i, lv_name in enumerate(final_lv_names):
        lv_idx = [j for j, v in enumerate(model.final_lvs) if v][i]

        x_colours = _meta_colours(x_meta_df, x_feature_names)
        plot_loadings(
            loadings=model.u_loadings[:, lv_idx],
            se=model.u_se[:, lv_idx],
            feature_names=x_feature_names,
            lv_name=lv_name,
            out_path=figures_dir / f"{lv_name}_x_loadings.{ext}",
            colours=x_colours,
            dpi=dpi,
        )

        y_colours = _meta_colours(y_meta_df, y_feature_names)
        plot_loadings(
            loadings=model.vt_loadings[lv_idx, :],
            se=model.vt_se[lv_idx, :],
            feature_names=y_feature_names,
            lv_name=lv_name,
            out_path=figures_dir / f"{lv_name}_y_loadings.{ext}",
            colours=y_colours,
            dpi=dpi,
        )

    # 4. Subject score box/strip plots
    if config is not None and len(config.groups) > 0 and len(final_lv_names) > 0:
        _plot_score_boxstrips(
            model, config, demo_aligned, subject_ids, sid,
            final_lv_names, figures_dir, ext, dpi,
        )

    # 5. Score scatter (correlational only)
    if method == "correlational" and config is not None and len(final_lv_names) > 0:
        _plot_score_scatters(
            model, config, demo_aligned, sid,
            final_lv_names, figures_dir, ext, dpi,
        )

    # --- Write log ---
    _write_log(output_dir, {
        "method": method,
        "x": str(x_path),
        "y": str(y_path),
        "demographics": str(demographics_path),
        "group_col": group_col,
        "groups": str(groups_path),
        "subject_id": sid,
        "n_perms": n_perms,
        "n_bootstraps": n_bootstraps,
        "seed": seed,
        "format": img_format,
        "dpi": dpi,
        "n_subjects": len(subject_ids),
        "n_x_features": len(x_feature_names),
        "n_y_features": len(y_feature_names),
        "significant_lvs": final_lv_names,
    })

    logger.info("PLS analysis complete. Results saved to: %s", output_dir)
    logger.info("Significant and reliable LVs: %s", final_lv_names)


def cross_validate_pipeline(
    *,
    y_path: Path,
    demographics_path: Path,
    output_dir: Path,
    group_col: str,
    subject_id: str | None = None,
    n_folds: int = 5,
    n_repeats: int = 100,
    n_components: int | None = None,
    n_permutations: int = 1000,
    seed: int = 42,
    img_format: str = "svg",
    dpi: int = 300,
) -> None:
    """Run cross-validation pipeline for discriminatory PLS.

    Parameters
    ----------
    y_path : Path
        Path to Y matrix CSV.
    demographics_path : Path
        Path to demographics CSV.
    output_dir : Path
        Output directory for results.
    group_col : str
        Group column name.
    subject_id : str, optional
        Subject ID column name.
    n_folds : int
        Number of CV folds.
    n_repeats : int
        Number of CV repeats.
    n_components : int, optional
        Number of PLS components (default: n_groups - 1).
    n_permutations : int
        Number of permutations for significance test.
    seed : int
        Random seed.
    img_format : str
        "svg" or "png".
    dpi : int
        Output image resolution.
    """
    from plsdo.cross_validate import run_cv, permutation_test_cv
    from plsdo.plotting import plot_cv_accuracy, plot_cv_permutation, plot_confusion_matrix

    # --- Set up output ---
    figures_dir = output_dir / "figures"
    data_dir_out = output_dir / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir_out.mkdir(parents=True, exist_ok=True)

    # --- Load and validate ---
    y_df = load_csv(y_path, require_numeric=True)
    demographics_df = load_csv(demographics_path)

    sid = detect_subject_id(
        [y_df, demographics_df], subject_id=subject_id
    )
    y_aligned, demo_aligned = align_subjects(
        [y_df, demographics_df], subject_id=sid
    )

    check_missing_values(y_aligned, name="Y")

    y_feature_names = [c for c in y_aligned.columns if c != sid]
    Y = y_aligned[y_feature_names].to_numpy(dtype=float)

    # Build group labels
    labels_series = demo_aligned[group_col].astype("category")
    label_names = labels_series.cat.categories.tolist()
    labels = labels_series.cat.codes.values
    n_groups = len(label_names)

    if n_components is None:
        n_components = n_groups - 1

    # --- Run CV ---
    logger.info("Running %d-fold CV with %d repeats...", n_folds, n_repeats)
    cv_result = run_cv(
        Y, labels,
        n_splits=n_folds, n_repeats=n_repeats,
        n_components=n_components, seed=seed,
    )

    logger.info("Mean accuracy: %.3f", cv_result["mean_accuracy"])
    logger.info("Chance level: %.3f", 1 / n_groups)

    # --- Permutation test ---
    logger.info("Running permutation test (%d permutations)...", n_permutations)
    perm_result = permutation_test_cv(
        Y, labels,
        observed_accuracy=cv_result["mean_accuracy"],
        n_splits=n_folds, n_repeats=1,
        n_components=n_components,
        n_permutations=n_permutations, seed=seed,
    )

    logger.info("Permutation p-value: %.4f", perm_result["p_value"])

    # --- Save data ---
    cv_result["fold_results"].to_csv(
        data_dir_out / "cv_fold_results.csv", index=False
    )
    pd.DataFrame({"null_accuracy": perm_result["null_accuracies"]}).to_csv(
        data_dir_out / "cv_permutation_accuracies.csv", index=False
    )

    # --- Plots ---
    ext = img_format
    chance = 1 / n_groups

    plot_cv_accuracy(
        fold_accuracies=cv_result["fold_results"]["accuracy"].values,
        mean_accuracy=cv_result["mean_accuracy"],
        chance_level=chance,
        out_path=figures_dir / f"cv_fold_accuracy.{ext}",
        dpi=dpi,
    )

    plot_cv_permutation(
        null_accuracies=perm_result["null_accuracies"],
        observed_accuracy=cv_result["mean_accuracy"],
        p_value=perm_result["p_value"],
        out_path=figures_dir / f"cv_permutation_test.{ext}",
        dpi=dpi,
    )

    plot_confusion_matrix(
        cm=cv_result["confusion_matrix"],
        label_names=label_names,
        mean_accuracy=cv_result["mean_accuracy"],
        out_path=figures_dir / f"cv_confusion_matrix.{ext}",
        dpi=dpi,
    )

    # --- Log ---
    _write_log(output_dir, {
        "command": "cross-validate",
        "y": str(y_path),
        "demographics": str(demographics_path),
        "group_col": group_col,
        "subject_id": sid,
        "n_folds": n_folds,
        "n_repeats": n_repeats,
        "n_components": n_components,
        "n_permutations": n_permutations,
        "seed": seed,
        "n_subjects": len(Y),
        "n_groups": n_groups,
        "group_names": label_names,
        "mean_accuracy": f"{cv_result['mean_accuracy']:.3f}",
        "permutation_p_value": f"{perm_result['p_value']:.4f}",
    })

    logger.info("Cross-validation complete. Results saved to: %s", output_dir)


# --- Private helpers ---


def _meta_colours(meta_df, feature_names):
    """Build per-feature colour list from metadata, or None."""
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


def _plot_score_boxstrips(
    model, config, demo_aligned, subject_ids, sid,
    final_lv_names, figures_dir, ext, dpi,
):
    """Build long-format score dataframes and produce box/strip plots."""
    group_cols_to_use = [g for g in config.groups if g.role != "ignore"]
    if not group_cols_to_use:
        return

    x_axis_col = next(
        (g.column for g in group_cols_to_use if g.role == "x_axis"),
        group_cols_to_use[0].column,
    )
    hue_col = next(
        (g.column for g in group_cols_to_use if g.role == "hue"),
        None,
    )

    for score_side, score_matrix in [("X", model.x_scores), ("Y", model.y_scores)]:
        score_long = []
        for i, lv_name in enumerate(final_lv_names):
            lv_idx = [j for j, v in enumerate(model.final_lvs) if v][i]
            for subj_i, subj_id in enumerate(subject_ids):
                row = {
                    sid: subj_id,
                    "score": score_matrix[subj_i, lv_idx],
                    "LV": lv_name,
                }
                for g in group_cols_to_use:
                    row[g.column] = demo_aligned.iloc[subj_i][g.column]
                score_long.append(row)

        score_long_df = pd.DataFrame(score_long)
        for g in group_cols_to_use:
            if g.order:
                score_long_df[g.column] = pd.Categorical(
                    score_long_df[g.column],
                    categories=g.order, ordered=True,
                )

        plot_scores_boxstrip(
            scores_df=score_long_df,
            x_col=x_axis_col,
            y_col="score",
            col_col="LV",
            hue_col=hue_col,
            out_path=figures_dir / f"{score_side}_scores_boxplot.{ext}",
            dpi=dpi,
        )


def _plot_score_scatters(
    model, config, demo_aligned, sid,
    final_lv_names, figures_dir, ext, dpi,
):
    """Produce score scatter plots (correlational PLS only)."""
    group_cols_to_use = [g for g in config.groups if g.role != "ignore"]
    hue_col = next(
        (g.column for g in group_cols_to_use if g.role in ("x_axis", "hue")),
        None,
    )
    for i, lv_name in enumerate(final_lv_names):
        lv_idx = [j for j, v in enumerate(model.final_lvs) if v][i]
        scatter_data = {
            "x_score": model.x_scores[:, lv_idx],
            "y_score": model.y_scores[:, lv_idx],
        }
        if hue_col:
            scatter_data[hue_col] = demo_aligned[hue_col].values
        scatter_df = pd.DataFrame(scatter_data)

        plot_scores_scatter(
            scatter_df=scatter_df,
            x_col="x_score", y_col="y_score",
            hue_col=hue_col or sid,
            lv_name=lv_name,
            out_path=figures_dir / f"{lv_name}_scores_scatter.{ext}",
            dpi=dpi,
        )
