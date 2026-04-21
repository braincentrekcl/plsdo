"""Input loading, validation, and preprocessing."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yaml
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


VALID_ROLES = {"x_axis", "hue", "facet_rows", "facet_cols", "ignore"}


@dataclass
class GroupSpec:
    """Specification for a single grouping column."""
    column: str
    role: str
    reference: Optional[str] = None
    order: Optional[list[str]] = None
    order_by: Optional[str] = None
    facet_col_wrap: Optional[int] = None


@dataclass
class GroupConfig:
    """Parsed groups configuration."""
    subject_id: Optional[str] = None
    groups: list[GroupSpec] = field(default_factory=list)

    @classmethod
    def from_group_col(cls, group_col: str, subject_id: Optional[str] = None):
        """Create a config from a single --group-col string."""
        return cls(
            subject_id=subject_id,
            groups=[GroupSpec(column=group_col, role="x_axis")],
        )


def parse_groups_config(
    yaml_path: Path,
    demographics_df: Optional[pd.DataFrame] = None,
) -> GroupConfig:
    """Parse a YAML groups configuration file.

    Parameters
    ----------
    yaml_path : Path
        Path to YAML file.
    demographics_df : DataFrame, optional
        If provided, validates that group columns exist in demographics
        and warns about unlisted columns.

    Returns
    -------
    GroupConfig
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    subject_id = raw.get("subject_id")
    groups = []

    for entry in raw.get("groups", []):
        role = entry.get("role", "")
        if role not in VALID_ROLES:
            raise ValueError(
                f"Invalid role '{role}' for column '{entry.get('column')}'. "
                f"Valid roles: {sorted(VALID_ROLES)}"
            )
        groups.append(GroupSpec(
            column=entry["column"],
            role=role,
            reference=entry.get("reference"),
            order=entry.get("order"),
            order_by=entry.get("order_by"),
            facet_col_wrap=entry.get("facet_col_wrap"),
        ))

    config = GroupConfig(subject_id=subject_id, groups=groups)

    # Validate against demographics if provided
    if demographics_df is not None:
        demo_cols = set(demographics_df.columns)
        listed_cols = {g.column for g in groups}

        for g in groups:
            if g.role != "ignore" and g.column not in demo_cols:
                raise ValueError(
                    f"Group column '{g.column}' not found in demographics. "
                    f"Available columns: {sorted(demo_cols)}"
                )

        # Warn about unlisted columns
        ignore_cols = {subject_id} if subject_id else set()
        unlisted = demo_cols - listed_cols - ignore_cols
        if unlisted:
            logger.warning(
                "Ignoring demographics columns not in groups config: %s",
                sorted(unlisted),
            )

    return config


def load_metadata(
    path: Path, data_feature_names: list[str]
) -> pd.DataFrame:
    """Load a feature metadata CSV and validate against data features.

    Parameters
    ----------
    path : Path
        Path to metadata CSV. Must have a 'feature' column.
    data_feature_names : list of str
        Feature names from the data matrix.

    Returns
    -------
    pd.DataFrame
    """
    df = load_csv(path)
    if "feature" not in df.columns:
        raise ValueError(
            f"Metadata file {path} must have a 'feature' column. "
            f"Columns found: {list(df.columns)}"
        )

    meta_features = set(df["feature"])
    data_features = set(data_feature_names)

    # Features in metadata but not in data: error
    extra = meta_features - data_features
    if extra:
        raise ValueError(
            f"Features in metadata but not in data: {sorted(extra)}. "
            f"Check for typos or stale metadata."
        )

    # Features in data but not in metadata: warning
    missing = data_features - meta_features
    if missing:
        logger.warning(
            "Features not in metadata (will not be colour-coded in plots): %s",
            sorted(missing),
        )

    return df


def build_design_matrix(
    demographics: pd.DataFrame, config: GroupConfig
) -> tuple[np.ndarray, list[str]]:
    """Build an additive dummy-coded design matrix from group columns.

    Parameters
    ----------
    demographics : DataFrame
        Demographics data with group columns.
    config : GroupConfig
        Groups configuration specifying which columns to use.

    Returns
    -------
    X : ndarray, shape (n_subjects, total_dummy_cols)
        Concatenated dummy codes for all non-ignore group columns.
    labels : list of str
        Column labels for X (e.g. ['geno_WT', 'geno_KO', 'drug_sal']).
    """
    all_dummies = []
    all_labels = []

    for spec in config.groups:
        if spec.role == "ignore":
            continue

        col = demographics[spec.column]
        levels = col.unique()

        if len(levels) < 2:
            raise ValueError(
                f"Group column '{spec.column}' has a single level "
                f"('{levels[0]}'). Need at least 2 levels for "
                f"discriminatory PLS."
            )

        # Determine level order
        if spec.order is not None:
            ordered_levels = spec.order
        elif spec.order_by is not None:
            order_df = (
                demographics[[spec.column, spec.order_by]]
                .drop_duplicates()
                .sort_values(spec.order_by)
            )
            ordered_levels = order_df[spec.column].tolist()
        elif spec.reference is not None:
            others = sorted([lev for lev in levels if lev != spec.reference])
            ordered_levels = [spec.reference] + others
        else:
            ordered_levels = sorted(levels)

        # Dummy code
        dummies = pd.get_dummies(col).reindex(columns=ordered_levels, fill_value=0)
        labels = [f"{spec.column}_{level}" for level in ordered_levels]

        all_dummies.append(dummies.to_numpy(dtype=float))
        all_labels.extend(labels)

    X = np.concatenate(all_dummies, axis=1)
    return X, all_labels
