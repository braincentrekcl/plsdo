"""Input loading, validation, and preprocessing."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import zscore

logger = logging.getLogger("plsdo")


def load_csv(path: Path, require_numeric: bool = False) -> pd.DataFrame:
    """Load a CSV file with validation.

    Parameters
    ----------
    path : Path
        Path to CSV file.
    require_numeric : bool
        If True, raise if no numeric columns found.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    ValueError
        If file is empty, unreadable, or has no numeric columns
        when required.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"File could not be read as CSV: {path} ({e})")

    if df.empty or len(df.columns) == 0:
        raise ValueError(f"File is empty: {path}")

    if require_numeric:
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            raise ValueError(
                f"File has no numeric columns: {path}. "
                f"Columns found: {list(df.columns)}"
            )

    return df


def detect_subject_id(
    dfs: list[pd.DataFrame], subject_id: Optional[str] = None
) -> str:
    """Identify the subject ID column across dataframes.

    Parameters
    ----------
    dfs : list of DataFrames
        All input dataframes to check.
    subject_id : str, optional
        Explicit column name. If None, auto-detect from shared columns.

    Returns
    -------
    str
        The subject ID column name.
    """
    if subject_id is not None:
        for i, df in enumerate(dfs):
            if subject_id not in df.columns:
                raise ValueError(
                    f"Subject ID column '{subject_id}' not found in "
                    f"dataframe {i} (columns: {list(df.columns)})"
                )
        return subject_id

    # Auto-detect: first column name shared across all dataframes
    shared = set(dfs[0].columns)
    for df in dfs[1:]:
        shared &= set(df.columns)

    if not shared:
        raise ValueError(
            "No shared column found across input files. "
            "Specify --subject-id explicitly."
        )

    # Pick the first shared column (alphabetically for determinism)
    sid = sorted(shared)[0]
    logger.info("Auto-detected subject ID column: '%s'", sid)
    return sid


def align_subjects(
    dfs: list[pd.DataFrame], subject_id: str
) -> list[pd.DataFrame]:
    """Align dataframes to shared subjects in consistent order.

    Parameters
    ----------
    dfs : list of DataFrames
        Input dataframes with a shared subject ID column.
    subject_id : str
        Name of the subject ID column.

    Returns
    -------
    list of DataFrames
        Reordered dataframes containing only shared subjects.
    """
    # Find intersection of subject IDs
    id_sets = [set(df[subject_id]) for df in dfs]
    shared_ids = id_sets[0]
    for s in id_sets[1:]:
        shared_ids &= s

    if not shared_ids:
        raise ValueError(
            "No subjects shared across all input files. "
            "Check that the subject ID column is correct."
        )

    # Warn about dropped subjects
    all_ids = set()
    for s in id_sets:
        all_ids |= s
    dropped = all_ids - shared_ids
    if dropped:
        logger.warning(
            "%d subject(s) not present in all files and will be excluded: %s",
            len(dropped), sorted(dropped),
        )

    # Reorder all dataframes to the same sorted subject order
    ordered_ids = sorted(shared_ids)
    aligned = []
    for df in dfs:
        reordered = (
            df[df[subject_id].isin(shared_ids)]
            .set_index(subject_id)
            .loc[ordered_ids]
            .reset_index()
        )
        aligned.append(reordered)

    return aligned


def check_missing_values(df: pd.DataFrame, name: str) -> None:
    """Check for NaN values in a dataframe and raise if found.

    Parameters
    ----------
    df : DataFrame
        Data to check (numeric columns only).
    name : str
        Name for error messages (e.g. "X", "Y").
    """
    numeric = df.select_dtypes(include="number")
    mask = numeric.isna()
    if mask.any().any():
        problem_features = mask.columns[mask.any()].tolist()
        problem_rows = mask.index[mask.any(axis=1)].tolist()
        raise ValueError(
            f"{name} has missing values. "
            f"Features with NaNs: {problem_features}. "
            f"Row indices with NaNs: {problem_rows}. "
            f"The module does not impute or drop data — fix the input files."
        )


def check_variance(
    arr: np.ndarray,
    feature_names: list[str],
    near_zero_threshold: float = 0.95,
) -> None:
    """Check for zero and near-zero variance features.

    Parameters
    ----------
    arr : ndarray, shape (n_subjects, n_features)
        Numeric data matrix.
    feature_names : list of str
        Feature names for error/warning messages.
    near_zero_threshold : float
        If this fraction of values in a column are identical, warn.
    """
    variances = np.var(arr, axis=0)

    # Zero variance is a hard error
    zero_var = np.where(variances == 0.0)[0]
    if len(zero_var) > 0:
        names = [feature_names[i] for i in zero_var]
        raise ValueError(
            f"Features with zero variance (cannot z-score): {names}. "
            f"Remove these features from the input file."
        )

    # Near-zero variance is a warning
    n_rows = arr.shape[0]
    for col_idx in range(arr.shape[1]):
        vals, counts = np.unique(arr[:, col_idx], return_counts=True)
        max_frac = counts.max() / n_rows
        if max_frac >= near_zero_threshold:
            logger.warning(
                "Feature '%s' has near-zero variance "
                "(%d%% of values identical). This may distort PLS results.",
                feature_names[col_idx], int(max_frac * 100),
            )


def zscore_columns(arr: np.ndarray) -> np.ndarray:
    """Z-score each column (feature) of a matrix.

    Parameters
    ----------
    arr : ndarray, shape (n_subjects, n_features)

    Returns
    -------
    ndarray
        Z-scored array, same shape.
    """
    return zscore(arr, axis=0, ddof=0)
