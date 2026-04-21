"""Input loading, validation, and preprocessing."""

from typing import Optional

import pandas as pd
from pathlib import Path


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
    print(f"INFO: Auto-detected subject ID column: '{sid}'")
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
        print(
            f"WARNING: {len(dropped)} subject(s) not present in all files "
            f"and will be excluded: {sorted(dropped)}"
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
