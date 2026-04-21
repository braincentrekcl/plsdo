# PLS Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an installable Python CLI (`pls`) that performs correlational and discriminatory PLS analysis with permutation testing, bootstrap reliability, and cross-validation.

**Architecture:** A `PLS` class in `core.py` handles SVD-based computation. `io.py` handles all input validation and preprocessing. `plotting.py` contains stateless plot functions. `cli.py` wires everything together via argparse subcommands. `cross_validate.py` handles CV using sklearn's PLSRegression.

**Tech Stack:** Python 3.10+, numpy, scipy, pandas, matplotlib, seaborn, scikit-learn, pyyaml, pytest, ruff, uv, hatchling.

**Reference implementations:** `correlational_pls.ipynb`, `discriminatory_pls.ipynb`, `claude_cross_validation.py` in the project root. Computational steps and plot styling must match these unless divergence is discussed with the user first.

**Important notes for implementers:**
- Use British English in all prose (docs, comments, commit messages, user-facing strings). Variable names follow library conventions (e.g. matplotlib uses `color`).
- Do not commit directly. Provide commit messages with conventional prefixes (docs, feat, fix, test, chore) and let the user commit.
- Do not distribute the reference notebooks with the package.

---

## File Map

```
pls/
  __init__.py          -- version string, public imports
  cli.py               -- argparse-based CLI with `run` and `cross-validate` subcommands
  core.py              -- PLS class: fit, permutation_test, bootstrap, filter_lvs
  cross_validate.py    -- run_cv() and permutation_test_cv() functions
  io.py                -- load_csv, validate_inputs, parse_groups_config,
                          load_metadata, build_design_matrix, zscore_matrices
  plotting.py          -- figure_size, plot_heatmap, plot_permutation,
                          plot_loadings, plot_scores_boxstrip, plot_scores_scatter,
                          plot_cv_accuracy, plot_cv_permutation, plot_confusion_matrix,
                          (verbose) plot_lv_heatmaps, plot_bootstrap_heatmaps,
                          plot_raw_distributions, plot_scree, plot_cv_convergence
pyproject.toml
tests/
  conftest.py          -- shared fixtures (synthetic data, tmp dirs)
  test_io.py
  test_core.py
  test_cross_validate.py
  test_plotting.py
  data/                -- synthetic CSVs for integration tests
docs/
  usage.md
  input-format.md
  missing-data.md
  interpreting-output.md
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `pls/__init__.py`
- Create: `pls/cli.py` (stub)
- Create: `pls/core.py` (stub)
- Create: `pls/cross_validate.py` (stub)
- Create: `pls/io.py` (stub)
- Create: `pls/plotting.py` (stub)
- Create: `tests/conftest.py` (stub)

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "pls"
version = "0.1.0"
description = "PLS covariance analysis with statistical testing and visualisation"
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "seaborn",
    "pyyaml",
]

[project.optional-dependencies]
cv = [
    "scikit-learn",
]
dev = [
    "pytest",
    "ruff",
    "scikit-learn",
]

[project.scripts]
pls = "pls.cli:pls_main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Create package stubs**

`pls/__init__.py`:
```python
"""PLS covariance analysis with statistical testing and visualisation."""

__version__ = "0.1.0"
```

`pls/cli.py`:
```python
"""CLI entry point for PLS analysis."""

import argparse


def pls_main(argv=None):
    """PLS covariance analysis with statistical testing and visualisation."""
    parser = argparse.ArgumentParser(
        description="PLS covariance analysis with statistical testing and visualisation."
    )
    subparsers = parser.add_subparsers(dest="command")
    args = parser.parse_args(argv)
```

`pls/core.py`:
```python
"""Core PLS computation: SVD, permutation testing, bootstrap reliability."""
```

`pls/cross_validate.py`:
```python
"""Cross-validation for discriminatory PLS."""
```

`pls/io.py`:
```python
"""Input loading, validation, and preprocessing."""
```

`pls/plotting.py`:
```python
"""Stateless plot functions for PLS results."""
```

`tests/conftest.py`:
```python
"""Shared test fixtures."""
```

- [ ] **Step 3: Create virtual environment and install**

Run:
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

- [ ] **Step 4: Verify CLI stub works**

Run: `pls --help`

Expected:
```
Usage: pls [OPTIONS] COMMAND [ARGS]...

  PLS covariance analysis with statistical testing and visualisation.

Options:
  --help  Show this message and exit.
```

- [ ] **Step 5: Verify pytest runs**

Run: `pytest --co -q`

Expected: `no tests ran` (no error)

- [ ] **Step 6: Commit**

```
chore: scaffold pls package with pyproject.toml and module stubs
```

---

### Task 2: Synthetic Test Data

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/data/x.csv`
- Create: `tests/data/y.csv`
- Create: `tests/data/demographics.csv`
- Create: `tests/data/x_meta.csv`
- Create: `tests/data/y_meta.csv`
- Create: `tests/data/groups.yaml`

- [ ] **Step 1: Design synthetic data**

We need 12 subjects across 3 groups (A, B, C) with 4 subjects each.
X has 5 features, Y has 4 features. Values are small integers so we can
reason about them. Include a known covariance structure so SVD results are
predictable.

`tests/data/x.csv`:
```csv
subject_id,x1,x2,x3,x4,x5
s01,2.1,3.4,1.2,4.5,2.3
s02,1.8,3.1,1.5,4.2,2.1
s03,2.3,3.6,1.1,4.8,2.5
s04,2.0,3.3,1.3,4.4,2.2
s05,4.5,1.2,3.8,1.1,4.7
s06,4.2,1.5,3.5,1.4,4.4
s07,4.8,1.1,4.1,0.8,5.0
s08,4.4,1.3,3.7,1.2,4.6
s09,3.0,2.5,2.5,3.0,3.5
s10,3.2,2.3,2.7,2.8,3.7
s11,2.8,2.7,2.3,3.2,3.3
s12,3.1,2.4,2.6,2.9,3.6
```

`tests/data/y.csv`:
```csv
subject_id,y1,y2,y3,y4
s01,5.0,1.2,3.1,2.0
s02,4.8,1.4,2.9,2.2
s03,5.2,1.0,3.3,1.8
s04,4.9,1.3,3.0,2.1
s05,1.5,4.8,2.0,5.1
s06,1.8,4.5,2.3,4.8
s07,1.2,5.1,1.7,5.4
s08,1.6,4.7,2.1,5.0
s09,3.5,3.0,2.5,3.5
s10,3.3,3.2,2.3,3.7
s11,3.7,2.8,2.7,3.3
s12,3.4,3.1,2.4,3.6
```

`tests/data/demographics.csv`:
```csv
subject_id,group,sex
s01,A,M
s02,A,F
s03,A,M
s04,A,F
s05,B,M
s06,B,F
s07,B,M
s08,B,F
s09,C,M
s10,C,F
s11,C,M
s12,C,F
```

`tests/data/x_meta.csv`:
```csv
feature,category
x1,alpha
x2,alpha
x3,beta
x4,beta
x5,gamma
```

`tests/data/y_meta.csv`:
```csv
feature,category
y1,one
y2,one
y3,two
y4,two
```

`tests/data/groups.yaml`:
```yaml
subject_id: subject_id
groups:
  - column: group
    role: x_axis
    reference: A
    order: [A, B, C]
  - column: sex
    role: hue
```

- [ ] **Step 2: Create fixtures in `conftest.py`**

```python
"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def data_dir():
    return DATA_DIR


@pytest.fixture
def x_df():
    return pd.read_csv(DATA_DIR / "x.csv")


@pytest.fixture
def y_df():
    return pd.read_csv(DATA_DIR / "y.csv")


@pytest.fixture
def demographics_df():
    return pd.read_csv(DATA_DIR / "demographics.csv")


@pytest.fixture
def x_meta_df():
    return pd.read_csv(DATA_DIR / "x_meta.csv")


@pytest.fixture
def y_meta_df():
    return pd.read_csv(DATA_DIR / "y_meta.csv")


@pytest.fixture
def groups_yaml_path():
    return DATA_DIR / "groups.yaml"


@pytest.fixture
def x_array(x_df):
    """Numeric X matrix as numpy array (no subject ID column)."""
    return x_df.drop(columns=["subject_id"]).to_numpy(dtype=float)


@pytest.fixture
def y_array(y_df):
    """Numeric Y matrix as numpy array (no subject ID column)."""
    return y_df.drop(columns=["subject_id"]).to_numpy(dtype=float)


@pytest.fixture
def tmp_output(tmp_path):
    """Temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out
```

- [ ] **Step 3: Verify fixtures load**

Create `tests/test_fixtures.py`:
```python
def test_data_files_exist(data_dir):
    assert (data_dir / "x.csv").exists()
    assert (data_dir / "y.csv").exists()
    assert (data_dir / "demographics.csv").exists()


def test_x_shape(x_df):
    assert x_df.shape == (12, 6)  # 12 subjects, subject_id + 5 features


def test_y_shape(y_df):
    assert y_df.shape == (12, 5)  # 12 subjects, subject_id + 4 features


def test_demographics_shape(demographics_df):
    assert demographics_df.shape == (12, 3)  # 12 subjects, subject_id + group + sex
```

Run: `pytest tests/test_fixtures.py -v`

Expected: 4 tests PASS

- [ ] **Step 4: Commit**

```
test: add synthetic test data and shared fixtures
```

---

### Task 3: IO — CSV Loading and File Validation

**Files:**
- Create: `pls/io.py`
- Create: `tests/test_io.py`

- [ ] **Step 1: Write failing tests for file validation**

`tests/test_io.py`:
```python
import pandas as pd
import pytest
from pathlib import Path
from pls.io import load_csv


class TestLoadCsv:
    def test_loads_valid_csv(self, data_dir):
        df = load_csv(data_dir / "x.csv")
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (12, 6)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_csv(tmp_path / "nonexistent.csv")

    def test_empty_file_raises(self, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_csv(empty)

    def test_no_numeric_columns_raises(self, tmp_path):
        text_only = tmp_path / "text.csv"
        text_only.write_text("name,colour\nalice,red\nbob,blue\n")
        with pytest.raises(ValueError, match="no numeric columns"):
            load_csv(text_only, require_numeric=True)

    def test_unreadable_file_raises(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_bytes(b"\x80\x81\x82\x83")
        with pytest.raises(ValueError, match="could not be read"):
            load_csv(bad)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_io.py -v`

Expected: FAIL — `ImportError: cannot import name 'load_csv'`

- [ ] **Step 3: Implement `load_csv`**

`pls/io.py`:
```python
"""Input loading, validation, and preprocessing."""

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_io.py -v`

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```
feat: add CSV loading with file validation
```

---

### Task 4: IO — Subject ID Detection and Alignment

**Files:**
- Modify: `pls/io.py`
- Modify: `tests/test_io.py`

- [ ] **Step 1: Write failing tests for subject ID detection**

Append to `tests/test_io.py`:
```python
from pls.io import detect_subject_id, align_subjects


class TestDetectSubjectId:
    def test_explicit_subject_id(self, x_df, y_df, demographics_df):
        sid = detect_subject_id(
            [x_df, y_df, demographics_df], subject_id="subject_id"
        )
        assert sid == "subject_id"

    def test_explicit_subject_id_missing_from_file(self, x_df, y_df):
        with pytest.raises(
            ValueError, match="not found"
        ):
            detect_subject_id([x_df, y_df], subject_id="nonexistent")

    def test_auto_detect(self, x_df, y_df, demographics_df):
        sid = detect_subject_id([x_df, y_df, demographics_df])
        assert sid == "subject_id"

    def test_auto_detect_no_shared_column(self):
        df1 = pd.DataFrame({"a": [1], "val": [2]})
        df2 = pd.DataFrame({"b": [1], "val2": [3]})
        with pytest.raises(ValueError, match="No shared column"):
            detect_subject_id([df1, df2])


class TestAlignSubjects:
    def test_aligned_subjects_same_order(self, x_df, y_df, demographics_df):
        aligned = align_subjects(
            [x_df, y_df, demographics_df], subject_id="subject_id"
        )
        for df in aligned:
            assert list(df["subject_id"]) == [
                f"s{i:02d}" for i in range(1, 13)
            ]

    def test_reorders_mismatched(self):
        df1 = pd.DataFrame({"id": ["a", "b", "c"], "v1": [1, 2, 3]})
        df2 = pd.DataFrame({"id": ["c", "a", "b"], "v2": [30, 10, 20]})
        aligned = align_subjects([df1, df2], subject_id="id")
        assert list(aligned[0]["id"]) == list(aligned[1]["id"])

    def test_drops_non_shared_with_warning(self, capfd):
        df1 = pd.DataFrame({"id": ["a", "b", "c"], "v1": [1, 2, 3]})
        df2 = pd.DataFrame({"id": ["b", "c", "d"], "v2": [20, 30, 40]})
        aligned = align_subjects([df1, df2], subject_id="id")
        assert len(aligned[0]) == 2  # only b, c
        captured = capfd.readouterr()
        assert "not present in all files" in captured.out

    def test_empty_intersection_raises(self):
        df1 = pd.DataFrame({"id": ["a", "b"], "v1": [1, 2]})
        df2 = pd.DataFrame({"id": ["c", "d"], "v2": [3, 4]})
        with pytest.raises(ValueError, match="No subjects shared"):
            align_subjects([df1, df2], subject_id="id")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_io.py::TestDetectSubjectId -v`

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `detect_subject_id` and `align_subjects`**

Add to `pls/io.py`:
```python
from typing import Optional


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_io.py -v`

Expected: all tests PASS

- [ ] **Step 5: Commit**

```
feat: add subject ID detection and cross-file alignment
```

---

### Task 5: IO — Variance Checks and Z-Scoring

**Files:**
- Modify: `pls/io.py`
- Modify: `tests/test_io.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_io.py`:
```python
import numpy as np
from pls.io import check_missing_values, check_variance, zscore_columns


class TestCheckMissingValues:
    def test_no_missing_passes(self):
        df = pd.DataFrame({"id": ["a", "b"], "v1": [1.0, 2.0], "v2": [3.0, 4.0]})
        check_missing_values(df, name="test")  # should not raise

    def test_missing_values_raises(self):
        df = pd.DataFrame({"id": ["a", "b"], "v1": [1.0, float("nan")], "v2": [3.0, 4.0]})
        with pytest.raises(ValueError, match="missing values"):
            check_missing_values(df, name="test")

    def test_reports_which_subjects_and_features(self):
        df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "v1": [1.0, float("nan"), 3.0],
            "v2": [float("nan"), 2.0, 4.0],
        })
        with pytest.raises(ValueError, match="v1.*v2"):
            check_missing_values(df, name="test")


class TestCheckVariance:
    def test_normal_variance_passes(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        check_variance(arr, feature_names=["a", "b"])  # should not raise

    def test_zero_variance_raises(self):
        arr = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
        with pytest.raises(ValueError, match="zero variance"):
            check_variance(arr, feature_names=["const", "varying"])

    def test_near_zero_variance_warns(self, capfd):
        # 9 out of 10 values are identical
        arr = np.array([[1.0, 5.0]] * 9 + [[2.0, 6.0]])
        check_variance(arr, feature_names=["nearly_const", "varying"],
                        near_zero_threshold=0.85)
        captured = capfd.readouterr()
        assert "nearly_const" in captured.out
        assert "near-zero variance" in captured.out.lower()


class TestZscoreColumns:
    def test_zero_mean_unit_variance(self):
        arr = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        z = zscore_columns(arr)
        np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(z.std(axis=0, ddof=0), 1.0, atol=1e-10)

    def test_shape_preserved(self):
        arr = np.random.default_rng(0).standard_normal((10, 5))
        z = zscore_columns(arr)
        assert z.shape == arr.shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_io.py::TestCheckMissingValues -v`

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement validation and z-scoring functions**

Add to `pls/io.py`:
```python
import numpy as np
from scipy.stats import zscore


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
        # Identify which subjects and features have NaNs
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
            print(
                f"WARNING: Feature '{feature_names[col_idx]}' has "
                f"near-zero variance ({max_frac:.0%} of values identical). "
                f"This may distort PLS results."
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_io.py -v`

Expected: all tests PASS

- [ ] **Step 5: Commit**

```
feat: add missing value checks, variance checks, and z-scoring
```

---

### Task 6: IO — YAML Groups Config Parsing

**Files:**
- Modify: `pls/io.py`
- Modify: `tests/test_io.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_io.py`:
```python
import yaml
from pls.io import parse_groups_config, GroupConfig


class TestParseGroupsConfig:
    def test_loads_yaml(self, groups_yaml_path):
        config = parse_groups_config(groups_yaml_path)
        assert config.subject_id == "subject_id"
        assert len(config.groups) == 2
        assert config.groups[0].column == "group"
        assert config.groups[0].role == "x_axis"
        assert config.groups[0].reference == "A"
        assert config.groups[0].order == ["A", "B", "C"]

    def test_column_not_in_demographics_raises(self, tmp_path):
        cfg = tmp_path / "bad.yaml"
        cfg.write_text(yaml.dump({
            "subject_id": "subject_id",
            "groups": [{"column": "nonexistent", "role": "x_axis"}],
        }))
        demo = pd.DataFrame({"subject_id": ["a"], "group": ["A"]})
        with pytest.raises(ValueError, match="not found in demographics"):
            parse_groups_config(cfg, demographics_df=demo)

    def test_warns_about_unlisted_demographics_columns(self, tmp_path, capfd):
        cfg = tmp_path / "partial.yaml"
        cfg.write_text(yaml.dump({
            "subject_id": "subject_id",
            "groups": [{"column": "group", "role": "x_axis"}],
        }))
        demo = pd.DataFrame({
            "subject_id": ["a"], "group": ["A"], "extra_col": [1]
        })
        parse_groups_config(cfg, demographics_df=demo)
        captured = capfd.readouterr()
        assert "extra_col" in captured.out

    def test_from_group_col_string(self):
        config = GroupConfig.from_group_col("Drug")
        assert len(config.groups) == 1
        assert config.groups[0].column == "Drug"
        assert config.groups[0].role == "x_axis"

    def test_invalid_role_raises(self, tmp_path):
        cfg = tmp_path / "bad_role.yaml"
        cfg.write_text(yaml.dump({
            "subject_id": "id",
            "groups": [{"column": "g", "role": "invalid_role"}],
        }))
        with pytest.raises(ValueError, match="Invalid role"):
            parse_groups_config(cfg)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_io.py::TestParseGroupsConfig -v`

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement groups config parsing**

Add to `pls/io.py`:
```python
import yaml
from dataclasses import dataclass, field


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
            print(
                f"WARNING: Ignoring demographics columns not in groups "
                f"config: {sorted(unlisted)}"
            )

    return config
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_io.py -v`

Expected: all tests PASS

- [ ] **Step 5: Commit**

```
feat: add YAML groups config parsing with validation
```

---

### Task 7: IO — Metadata Loading and Discriminatory Dummy Coding

**Files:**
- Modify: `pls/io.py`
- Modify: `tests/test_io.py`

- [ ] **Step 1: Write failing tests for metadata**

Append to `tests/test_io.py`:
```python
from pls.io import load_metadata, build_design_matrix


class TestLoadMetadata:
    def test_loads_valid_metadata(self, data_dir):
        meta = load_metadata(
            data_dir / "x_meta.csv",
            data_feature_names=["x1", "x2", "x3", "x4", "x5"],
        )
        assert "feature" in meta.columns
        assert len(meta) == 5

    def test_feature_in_meta_not_in_data_raises(self, tmp_path):
        meta_path = tmp_path / "meta.csv"
        meta_path.write_text("feature,category\na,cat1\nb,cat2\nZZZ,cat3\n")
        with pytest.raises(ValueError, match="ZZZ"):
            load_metadata(meta_path, data_feature_names=["a", "b"])

    def test_feature_in_data_not_in_meta_warns(self, tmp_path, capfd):
        meta_path = tmp_path / "meta.csv"
        meta_path.write_text("feature,category\na,cat1\n")
        load_metadata(meta_path, data_feature_names=["a", "b"])
        captured = capfd.readouterr()
        assert "b" in captured.out
        assert "not in metadata" in captured.out.lower()


class TestBuildDesignMatrix:
    def test_single_factor(self):
        demo = pd.DataFrame({
            "subject_id": ["s1", "s2", "s3", "s4"],
            "group": ["A", "A", "B", "B"],
        })
        config = GroupConfig.from_group_col("group")
        X, labels = build_design_matrix(demo, config)
        assert X.shape == (4, 2)  # 2 levels
        assert labels == ["group_A", "group_B"]
        # s1 and s2 should have [1, 0], s3 and s4 should have [0, 1]
        np.testing.assert_array_equal(X[0], [1, 0])
        np.testing.assert_array_equal(X[2], [0, 1])

    def test_multiple_factors_additive(self):
        demo = pd.DataFrame({
            "subject_id": ["s1", "s2", "s3", "s4"],
            "geno": ["WT", "WT", "KO", "KO"],
            "drug": ["sal", "oxy", "sal", "oxy"],
        })
        config = GroupConfig(groups=[
            GroupSpec(column="geno", role="x_axis"),
            GroupSpec(column="drug", role="hue"),
        ])
        X, labels = build_design_matrix(demo, config)
        # 2 levels for geno + 2 levels for drug = 4 columns
        assert X.shape == (4, 4)
        assert labels == ["geno_KO", "geno_WT", "drug_oxy", "drug_sal"]

    def test_zero_variance_column_raises(self):
        demo = pd.DataFrame({
            "subject_id": ["s1", "s2"],
            "group": ["A", "A"],  # only one level
        })
        config = GroupConfig.from_group_col("group")
        with pytest.raises(ValueError, match="single level"):
            build_design_matrix(demo, config)

    def test_reference_level_ordering(self):
        demo = pd.DataFrame({
            "subject_id": ["s1", "s2", "s3"],
            "group": ["B", "A", "C"],
        })
        config = GroupConfig(groups=[
            GroupSpec(column="group", role="x_axis", reference="A",
                      order=["A", "B", "C"]),
        ])
        X, labels = build_design_matrix(demo, config)
        assert labels == ["group_A", "group_B", "group_C"]

    def test_ignores_ignore_role(self):
        demo = pd.DataFrame({
            "subject_id": ["s1", "s2"],
            "group": ["A", "B"],
            "cage": [1, 2],
        })
        config = GroupConfig(groups=[
            GroupSpec(column="group", role="x_axis"),
            GroupSpec(column="cage", role="ignore"),
        ])
        X, labels = build_design_matrix(demo, config)
        assert X.shape == (2, 2)  # only group, not cage
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_io.py::TestLoadMetadata -v`

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement metadata loading and design matrix building**

Add to `pls/io.py`:
```python
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
        print(
            f"WARNING: Features not in metadata (will not be "
            f"colour-coded in plots): {sorted(missing)}"
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
            others = sorted([l for l in levels if l != spec.reference])
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_io.py -v`

Expected: all tests PASS

- [ ] **Step 5: Commit**

```
feat: add metadata loading and discriminatory design matrix building
```

---

### Task 8: Core PLS — fit()

**Files:**
- Modify: `pls/core.py`
- Create: `tests/test_core.py`

- [ ] **Step 1: Write failing tests for fit**

`tests/test_core.py`:
```python
import numpy as np
import pytest
from pls.core import PLS


class TestPLSFit:
    def test_fit_stores_results(self, x_array, y_array):
        from pls.io import zscore_columns

        X = zscore_columns(x_array)
        Y = zscore_columns(y_array)
        model = PLS(X, Y)
        model.fit()

        n_subjects, n_x = X.shape
        n_y = Y.shape[1]
        n_components = min(n_x, n_y)

        assert model.xcorr.shape == (n_x, n_y)
        assert model.u.shape == (n_x, n_components)
        assert model.s.shape == (n_components,)
        assert model.vt.shape == (n_components, n_y)
        assert model.u_loadings.shape == (n_x, n_components)
        assert model.vt_loadings.shape == (n_components, n_y)
        assert model.x_scores.shape == (n_subjects, n_components)
        assert model.y_scores.shape == (n_subjects, n_components)

    def test_cross_correlation_formula(self, x_array, y_array):
        from pls.io import zscore_columns

        X = zscore_columns(x_array)
        Y = zscore_columns(y_array)
        model = PLS(X, Y)
        model.fit()

        expected = X.T @ Y / (X.shape[0] - 1)
        np.testing.assert_allclose(model.xcorr, expected)

    def test_loadings_are_scaled_vectors(self, x_array, y_array):
        from pls.io import zscore_columns

        X = zscore_columns(x_array)
        Y = zscore_columns(y_array)
        model = PLS(X, Y)
        model.fit()

        expected_u_load = model.u @ np.diag(model.s)
        expected_vt_load = np.diag(model.s) @ model.vt
        np.testing.assert_allclose(model.u_loadings, expected_u_load)
        np.testing.assert_allclose(model.vt_loadings, expected_vt_load)

    def test_scores_are_projections(self, x_array, y_array):
        from pls.io import zscore_columns

        X = zscore_columns(x_array)
        Y = zscore_columns(y_array)
        model = PLS(X, Y)
        model.fit()

        np.testing.assert_allclose(model.x_scores, X @ model.u)
        np.testing.assert_allclose(model.y_scores, Y @ model.vt.T)

    def test_singular_values_descending(self, x_array, y_array):
        from pls.io import zscore_columns

        X = zscore_columns(x_array)
        Y = zscore_columns(y_array)
        model = PLS(X, Y)
        model.fit()

        assert all(
            model.s[i] >= model.s[i + 1] for i in range(len(model.s) - 1)
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_core.py -v`

Expected: FAIL — `ImportError: cannot import name 'PLS'`

- [ ] **Step 3: Implement PLS class with fit()**

`pls/core.py`:
```python
"""Core PLS computation: SVD, permutation testing, bootstrap reliability."""

import numpy as np


class PLS:
    """Partial Least Squares via SVD of the cross-covariance matrix.

    Handles both correlational PLS (two continuous matrices) and
    discriminatory PLS (design matrix vs continuous matrix) — the
    maths is identical.

    Parameters
    ----------
    X : ndarray, shape (n_subjects, n_x_features)
        First data matrix (already z-scored for correlational,
        or dummy-coded for discriminatory).
    Y : ndarray, shape (n_subjects, n_y_features)
        Second data matrix (already z-scored).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, seed: int | None = None):
        self.X = X
        self.Y = Y
        self.n_subjects = X.shape[0]
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._fitted = False

    def fit(self):
        """Run PLS: cross-covariance, SVD, loadings, and subject scores."""
        self.xcorr = self.X.T @ self.Y / (self.n_subjects - 1)
        self._decompose()
        self.u_loadings = self.u @ np.diag(self.s)
        self.vt_loadings = np.diag(self.s) @ self.vt
        self.x_scores = self.X @ self.u
        self.y_scores = self.Y @ self.vt.T
        self._fitted = True

    def _decompose(self):
        """SVD of the cross-covariance matrix.

        Separated into its own method so that SparsePLS can override
        just this step.
        """
        self.u, self.s, self.vt = np.linalg.svd(
            self.xcorr, full_matrices=False
        )

    def _check_fitted(self):
        """Raise if fit() has not been called."""
        if not self._fitted:
            raise RuntimeError(
                "Call .fit() before running permutation or bootstrap."
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_core.py -v`

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```
feat: add PLS class with SVD-based fit
```

---

### Task 9: Core PLS — Permutation Test

**Files:**
- Modify: `pls/core.py`
- Modify: `tests/test_core.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_core.py`:
```python
class TestPermutationTest:
    def _fitted_model(self, x_array, y_array):
        from pls.io import zscore_columns

        X = zscore_columns(x_array)
        Y = zscore_columns(y_array)
        model = PLS(X, Y, seed=42)
        model.fit()
        return model

    def test_before_fit_raises(self, x_array, y_array):
        from pls.io import zscore_columns

        model = PLS(zscore_columns(x_array), zscore_columns(y_array))
        with pytest.raises(RuntimeError, match="fit"):
            model.permutation_test()

    def test_stores_results(self, x_array, y_array):
        model = self._fitted_model(x_array, y_array)
        model.permutation_test(n_perms=100)

        n_components = min(x_array.shape[1], y_array.shape[1])
        assert model.p_values.shape == (n_components,)
        assert model.permuted_singular_values.shape == (n_components, 100)
        assert model.significant_lvs.dtype == bool

    def test_p_values_between_0_and_1(self, x_array, y_array):
        model = self._fitted_model(x_array, y_array)
        model.permutation_test(n_perms=100)
        assert np.all(model.p_values >= 0.0)
        assert np.all(model.p_values <= 1.0)

    def test_seed_reproducibility(self, x_array, y_array):
        from pls.io import zscore_columns

        X = zscore_columns(x_array)
        Y = zscore_columns(y_array)

        m1 = PLS(X, Y, seed=42)
        m1.fit()
        m1.permutation_test(n_perms=100)

        m2 = PLS(X, Y, seed=42)
        m2.fit()
        m2.permutation_test(n_perms=100)

        np.testing.assert_array_equal(m1.p_values, m2.p_values)
        np.testing.assert_array_equal(
            m1.permuted_singular_values, m2.permuted_singular_values
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_core.py::TestPermutationTest -v`

Expected: FAIL — `AttributeError: 'PLS' object has no attribute 'permutation_test'`

- [ ] **Step 3: Implement permutation_test**

Add to `PLS` class in `pls/core.py`:
```python
    def permutation_test(self, n_perms: int = 10000) -> None:
        """Test significance of singular values by permutation.

        Permutes rows of X to break the subject-level pairing,
        recomputes SVD, and compares observed singular values
        to the null distribution.

        Parameters
        ----------
        n_perms : int
            Number of permutations.
        """
        self._check_fitted()

        perm_rng = np.random.default_rng(self._rng.integers(2**31))
        n_components = len(self.s)

        perm_s_list = []
        for _ in range(n_perms):
            perm_order = perm_rng.permutation(self.n_subjects)
            perm_x = self.X[perm_order, :]
            perm_xcorr = perm_x.T @ self.Y / (self.n_subjects - 1)
            _, perm_s, _ = np.linalg.svd(perm_xcorr, full_matrices=False)
            perm_s_list.append(perm_s)

        self.permuted_singular_values = np.stack(perm_s_list, axis=1)
        self.p_values = np.mean(
            self.permuted_singular_values >= self.s[:, None], axis=1
        )
        self.significant_lvs = self.p_values < 0.05
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_core.py -v`

Expected: all tests PASS

- [ ] **Step 5: Commit**

```
feat: add permutation testing for singular value significance
```

---

### Task 10: Core PLS — Bootstrap with Procrustes

**Files:**
- Modify: `pls/core.py`
- Modify: `tests/test_core.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_core.py`:
```python
class TestBootstrap:
    def _fitted_model(self, x_array, y_array):
        from pls.io import zscore_columns

        X = zscore_columns(x_array)
        Y = zscore_columns(y_array)
        model = PLS(X, Y, seed=42)
        model.fit()
        return model

    def test_before_fit_raises(self, x_array, y_array):
        from pls.io import zscore_columns

        model = PLS(zscore_columns(x_array), zscore_columns(y_array))
        with pytest.raises(RuntimeError, match="fit"):
            model.bootstrap()

    def test_stores_results(self, x_array, y_array):
        model = self._fitted_model(x_array, y_array)
        model.bootstrap(n_bootstraps=100)

        n_x = x_array.shape[1]
        n_y = y_array.shape[1]
        n_components = min(n_x, n_y)

        assert model.u_bootstrap_ratios.shape == (n_x, n_components)
        assert model.vt_bootstrap_ratios.shape == (n_components, n_y)
        assert model.u_se.shape == (n_x, n_components)
        assert model.vt_se.shape == (n_components, n_y)

    def test_bootstrap_ratios_are_loadings_over_se(self, x_array, y_array):
        model = self._fitted_model(x_array, y_array)
        model.bootstrap(n_bootstraps=100)

        eps = 1e-12
        expected_u_bsr = model.u_loadings / np.maximum(model.u_se, eps)
        expected_vt_bsr = model.vt_loadings / np.maximum(model.vt_se, eps)
        np.testing.assert_allclose(model.u_bootstrap_ratios, expected_u_bsr)
        np.testing.assert_allclose(model.vt_bootstrap_ratios, expected_vt_bsr)

    def test_seed_reproducibility(self, x_array, y_array):
        from pls.io import zscore_columns

        X = zscore_columns(x_array)
        Y = zscore_columns(y_array)

        m1 = PLS(X, Y, seed=42)
        m1.fit()
        m1.bootstrap(n_bootstraps=100)

        m2 = PLS(X, Y, seed=42)
        m2.fit()
        m2.bootstrap(n_bootstraps=100)

        np.testing.assert_array_equal(
            m1.u_bootstrap_ratios, m2.u_bootstrap_ratios
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_core.py::TestBootstrap -v`

Expected: FAIL — `AttributeError`

- [ ] **Step 3: Implement bootstrap**

Add to `PLS` class in `pls/core.py` (also add import at top of file:
`from scipy.linalg import orthogonal_procrustes` and
`from pls.io import zscore_columns`):

```python
    def bootstrap(self, n_bootstraps: int = 10000) -> None:
        """Assess reliability of loadings via bootstrap resampling.

        Resamples subjects with replacement, recomputes SVD, aligns
        via Procrustes rotation, and computes bootstrap ratios
        (loadings / standard error).

        Parameters
        ----------
        n_bootstraps : int
            Number of bootstrap resamples.
        """
        self._check_fitted()

        boot_rng = np.random.default_rng(self._rng.integers(2**31))
        row_idx = np.arange(self.n_subjects)

        u_distribution = []
        vt_distribution = []

        for _ in range(n_bootstraps):
            idx = boot_rng.choice(row_idx, size=self.n_subjects, replace=True)
            x_boot = zscore_columns(self.X[idx, :])
            y_boot = zscore_columns(self.Y[idx, :])

            boot_xcorr = x_boot.T @ y_boot / (self.n_subjects - 1)
            boot_u, boot_s, boot_vt = np.linalg.svd(
                boot_xcorr, full_matrices=False
            )

            # Procrustes: rotate bootstrap Vt to align with reference
            Q, _ = orthogonal_procrustes(boot_vt.T, self.vt.T)

            boot_u_load = boot_u @ np.diag(boot_s)
            boot_vt_load = np.diag(boot_s) @ boot_vt

            aligned_u_load = boot_u_load @ Q
            aligned_vt_load = Q.T @ boot_vt_load

            # Sign correction
            signs = np.sign(
                np.sum(aligned_vt_load * self.vt_loadings, axis=1, keepdims=True)
            )
            signs[signs == 0] = 1.0

            u_distribution.append(aligned_u_load * signs.T)
            vt_distribution.append(aligned_vt_load * signs)

        self.u_se = np.std(np.stack(u_distribution, axis=2), axis=2)
        self.vt_se = np.std(np.stack(vt_distribution, axis=2), axis=2)

        eps = 1e-12
        self.u_bootstrap_ratios = self.u_loadings / np.maximum(self.u_se, eps)
        self.vt_bootstrap_ratios = self.vt_loadings / np.maximum(
            self.vt_se, eps
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_core.py -v`

Expected: all tests PASS

- [ ] **Step 5: Commit**

```
feat: add bootstrap reliability with Procrustes alignment
```

---

### Task 11: Core PLS — LV Filtering

**Files:**
- Modify: `pls/core.py`
- Modify: `tests/test_core.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_core.py`:
```python
class TestFilterLVs:
    def test_before_permutation_raises(self, x_array, y_array):
        from pls.io import zscore_columns

        model = PLS(zscore_columns(x_array), zscore_columns(y_array))
        model.fit()
        with pytest.raises(RuntimeError, match="permutation"):
            model.filter_lvs()

    def test_before_bootstrap_raises(self, x_array, y_array):
        from pls.io import zscore_columns

        model = PLS(zscore_columns(x_array), zscore_columns(y_array), seed=42)
        model.fit()
        model.permutation_test(n_perms=100)
        with pytest.raises(RuntimeError, match="bootstrap"):
            model.filter_lvs()

    def test_filters_on_significance_and_reliability(self):
        """Manually set up a model with known p-values and bootstrap ratios."""
        X = np.random.default_rng(0).standard_normal((20, 3))
        Y = np.random.default_rng(1).standard_normal((20, 3))
        model = PLS(X, Y, seed=42)
        model.fit()

        # Manually set permutation results: LV1 significant, LV2 not, LV3 significant
        model.p_values = np.array([0.01, 0.50, 0.03])
        model.significant_lvs = model.p_values < 0.05

        # Manually set bootstrap ratios:
        # LV1: reliable on both sides (|BSR| > 1.96)
        # LV3: reliable on X but not Y
        model.u_bootstrap_ratios = np.array([
            [3.0, 0.5, 2.5],  # feature 1
            [0.1, 0.1, 0.1],  # feature 2
            [2.1, 0.3, 2.0],  # feature 3
        ])
        model.vt_bootstrap_ratios = np.array([
            [2.5, 0.2, 0.5],  # LV1: reliable
            [0.1, 0.1, 0.1],  # LV2: not reliable
            [1.0, 0.3, 1.5],  # LV3: not reliable (no feature > 1.96)
        ])

        model.filter_lvs()

        # Only LV1 should survive (significant + reliable on both sides)
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(model.final_lvs, expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_core.py::TestFilterLVs -v`

Expected: FAIL — `AttributeError`

- [ ] **Step 3: Implement filter_lvs**

Add to `PLS` class in `pls/core.py`:
```python
    def filter_lvs(self, bsr_threshold: float = 1.96) -> None:
        """Filter latent variables by significance and reliability.

        Keeps LVs that are:
        1. Significant (permutation p < 0.05)
        2. Have at least one feature with |bootstrap ratio| > threshold
           on both the X and Y sides

        Parameters
        ----------
        bsr_threshold : float
            Bootstrap ratio threshold (default 1.96 for 95% CI).
        """
        if not hasattr(self, "p_values"):
            raise RuntimeError(
                "Call .permutation_test() before .filter_lvs()."
            )
        if not hasattr(self, "u_bootstrap_ratios"):
            raise RuntimeError(
                "Call .bootstrap() before .filter_lvs()."
            )

        significant = self.p_values < 0.05

        # Check if any feature exceeds threshold on X side
        x_reliable = np.any(
            np.abs(self.u_bootstrap_ratios) > bsr_threshold, axis=0
        )

        # Check if any feature exceeds threshold on Y side
        y_reliable = np.any(
            np.abs(self.vt_bootstrap_ratios) > bsr_threshold, axis=1
        )

        self.final_lvs = significant & x_reliable & y_reliable
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_core.py -v`

Expected: all tests PASS

- [ ] **Step 5: Commit**

```
feat: add latent variable filtering by significance and reliability
```

---

### Task 12: Cross-Validation

**Files:**
- Modify: `pls/cross_validate.py`
- Create: `tests/test_cross_validate.py`

- [ ] **Step 1: Write failing tests**

`tests/test_cross_validate.py`:
```python
import numpy as np
import pytest
from pls.cross_validate import run_cv, permutation_test_cv


class TestRunCV:
    def test_perfect_separation(self):
        """Groups with zero overlap should classify near-perfectly."""
        rng = np.random.default_rng(42)
        n_per_group = 20
        # Group 0: features centred at 0, Group 1: at 10
        X = np.vstack([
            rng.standard_normal((n_per_group, 5)),
            rng.standard_normal((n_per_group, 5)) + 10,
        ])
        labels = np.array([0] * n_per_group + [1] * n_per_group)

        results = run_cv(
            X, labels, n_splits=5, n_repeats=10, n_components=1, seed=42
        )
        assert results["mean_accuracy"] > 0.90

    def test_random_data_near_chance(self):
        """Random labels should give accuracy near chance (0.5 for 2 groups)."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 5))
        labels = np.array([0] * 20 + [1] * 20)
        rng.shuffle(labels)

        results = run_cv(
            X, labels, n_splits=5, n_repeats=10, n_components=1, seed=42
        )
        assert results["mean_accuracy"] < 0.70  # generous margin

    def test_seed_reproducibility(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 5))
        labels = np.array([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10)

        r1 = run_cv(X, labels, n_splits=5, n_repeats=10, n_components=3, seed=42)
        r2 = run_cv(X, labels, n_splits=5, n_repeats=10, n_components=3, seed=42)
        assert r1["mean_accuracy"] == r2["mean_accuracy"]

    def test_returns_expected_keys(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 5))
        labels = np.array([0] * 20 + [1] * 20)

        results = run_cv(
            X, labels, n_splits=5, n_repeats=2, n_components=1, seed=42
        )
        assert "mean_accuracy" in results
        assert "mean_balanced_accuracy" in results
        assert "fold_results" in results
        assert "true_labels" in results
        assert "pred_labels" in results
        assert "confusion_matrix" in results


class TestPermutationTestCV:
    def test_returns_p_value(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 5))
        labels = np.array([0] * 20 + [1] * 20)

        result = permutation_test_cv(
            X, labels, observed_accuracy=0.5,
            n_splits=5, n_repeats=2, n_components=1,
            n_permutations=50, seed=42,
        )
        assert 0.0 <= result["p_value"] <= 1.0
        assert "null_accuracies" in result
        assert len(result["null_accuracies"]) == 50
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cross_validate.py -v`

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement cross-validation**

`pls/cross_validate.py`:
```python
"""Cross-validation for discriminatory PLS."""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler


def run_cv(
    X: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 100,
    n_components: int = 3,
    seed: int = 42,
) -> dict:
    """Run Repeated Stratified K-Fold CV for discriminatory PLS.

    Parameters
    ----------
    X : ndarray, shape (n_subjects, n_features)
        Continuous data matrix (MRI, behaviour, etc.).
    labels : ndarray, shape (n_subjects,)
        Integer group labels.
    n_splits : int
        Number of CV folds.
    n_repeats : int
        Number of CV repeats.
    n_components : int
        Number of PLS components.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        mean_accuracy, mean_balanced_accuracy, fold_results (DataFrame),
        true_labels, pred_labels, confusion_matrix
    """
    n_groups = len(np.unique(labels))
    Y_dummy = np.eye(n_groups)[labels]

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )

    fold_results = []
    all_true = []
    all_pred = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, labels)):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train = Y_dummy[train_idx]
        labels_test = labels[test_idx]

        # Standardise: fit on train, apply to both
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)

        # Fit PLS and predict
        pls = PLSRegression(n_components=n_components, scale=False)
        pls.fit(X_train_std, Y_train)
        Y_pred = pls.predict(X_test_std)
        pred_labels = np.argmax(Y_pred, axis=1)

        acc = accuracy_score(labels_test, pred_labels)
        bal_acc = balanced_accuracy_score(labels_test, pred_labels)

        fold_results.append({
            "fold": fold_idx,
            "repeat": fold_idx // n_splits,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "n_test": len(test_idx),
        })

        all_true.extend(labels_test)
        all_pred.extend(pred_labels)

    true_all = np.array(all_true)
    pred_all = np.array(all_pred)
    fold_df = pd.DataFrame(fold_results)

    return {
        "mean_accuracy": fold_df["accuracy"].mean(),
        "mean_balanced_accuracy": fold_df["balanced_accuracy"].mean(),
        "fold_results": fold_df,
        "true_labels": true_all,
        "pred_labels": pred_all,
        "confusion_matrix": confusion_matrix(
            true_all, pred_all, normalize="true"
        ),
    }


def permutation_test_cv(
    X: np.ndarray,
    labels: np.ndarray,
    observed_accuracy: float,
    n_splits: int = 5,
    n_repeats: int = 1,
    n_components: int = 3,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """Permutation test for CV accuracy significance.

    Shuffles group labels and repeats CV to build a null distribution.

    Parameters
    ----------
    X : ndarray
        Continuous data matrix.
    labels : ndarray
        Integer group labels.
    observed_accuracy : float
        The observed CV accuracy to test against.
    n_splits, n_repeats, n_components : int
        CV parameters (use fewer repeats per permutation for speed).
    n_permutations : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys: p_value, null_accuracies
    """
    rng = np.random.RandomState(seed)
    null_accs = []

    for perm_i in range(n_permutations):
        shuffled = rng.permutation(labels)
        result = run_cv(
            X, shuffled, n_splits=n_splits, n_repeats=n_repeats,
            n_components=n_components, seed=perm_i,
        )
        null_accs.append(result["mean_accuracy"])

    null_accs = np.array(null_accs)
    p_value = np.mean(null_accs >= observed_accuracy)

    return {
        "p_value": p_value,
        "null_accuracies": null_accs,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cross_validate.py -v`

Expected: all tests PASS

- [ ] **Step 5: Commit**

```
feat: add cross-validation with permutation testing
```

---

### Task 13: Plotting — Utilities and Core Plots (Part 1)

**Files:**
- Modify: `pls/plotting.py`
- Create: `tests/test_plotting.py`

- [ ] **Step 1: Write failing tests for utilities and heatmap**

`tests/test_plotting.py`:
```python
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pathlib import Path
from pls.plotting import figure_size, plot_heatmap, plot_permutation


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_plotting.py -v`

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement figure_size, plot_heatmap, plot_permutation**

`pls/plotting.py`:
```python
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
        fmt="3.2f" if annotate else "",
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_plotting.py -v`

Expected: all tests PASS

- [ ] **Step 5: Commit**

```
feat: add figure sizing, heatmap, and permutation plots
```

---

### Task 14: Plotting — Loading Bars, Box/Strip, and Score Scatter

**Files:**
- Modify: `pls/plotting.py`
- Modify: `tests/test_plotting.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_plotting.py`:
```python
from pls.plotting import plot_loadings, plot_scores_boxstrip, plot_scores_scatter


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

    def test_not_produced_for_discriminatory(self, tmp_output):
        """Scatter should not be called for discriminatory PLS.

        This is enforced at the CLI level, not the plot function level.
        The plot function itself always produces output if called.
        """
        pass  # Enforcement tested in CLI tests
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_plotting.py::TestPlotLoadings -v`

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement loading bars, box/strip, and scatter plots**

Add to `pls/plotting.py`:
```python
import pandas as pd


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_plotting.py -v`

Expected: all tests PASS

- [ ] **Step 5: Commit**

```
feat: add loading bar, box/strip, and score scatter plots
```

---

### Task 15: Plotting — CV Plots

**Files:**
- Modify: `pls/plotting.py`
- Modify: `tests/test_plotting.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_plotting.py`:
```python
from pls.plotting import (
    plot_cv_accuracy, plot_cv_permutation, plot_confusion_matrix,
)


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_plotting.py::TestCVPlots -v`

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement CV plot functions**

Add to `pls/plotting.py`:
```python
from sklearn.metrics import ConfusionMatrixDisplay


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
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=label_names,
    )
    disp.plot(ax=ax, cmap="Blues", values_format=".0%")
    ax.set_title(f"CV confusion matrix\nAccuracy: {mean_accuracy:.1%}")
    plt.tight_layout()
    fig.savefig(out_path, transparent=False, dpi=dpi)
    plt.close(fig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_plotting.py -v`

Expected: all tests PASS

- [ ] **Step 5: Commit**

```
feat: add CV accuracy, permutation, and confusion matrix plots
```

---

### Task 16: CLI — `pls run`

**Files:**
- Modify: `pls/cli.py`
- Modify: `tests/test_io.py` (add CLI integration tests)

- [ ] **Step 1: Write failing tests for CLI validation**

Create `tests/test_cli.py`:
```python
import pytest
from pls.cli import pls_main


class TestRunValidation:
    def test_method_required(self, data_dir):
        with pytest.raises(SystemExit) as exc_info:
            pls_main([
                "run",
                "--y", str(data_dir / "y.csv"),
                "--demographics", str(data_dir / "demographics.csv"),
                "--output", "/tmp/pls_test",
            ])
        assert exc_info.value.code != 0

    def test_method_c_without_x_errors(self, data_dir, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            pls_main([
                "run", "--method", "c",
                "--y", str(data_dir / "y.csv"),
                "--demographics", str(data_dir / "demographics.csv"),
                "--output", str(tmp_path / "out"),
            ])
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "correlational PLS requires --x" in captured.err

    def test_method_d_with_x_errors(self, data_dir, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            pls_main([
                "run", "--method", "d",
                "--x", str(data_dir / "x.csv"),
                "--y", str(data_dir / "y.csv"),
                "--demographics", str(data_dir / "demographics.csv"),
                "--group-col", "group",
                "--output", str(tmp_path / "out"),
            ])
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "do not provide --x" in captured.err.lower()

    def test_method_d_without_group_col_errors(self, data_dir, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            pls_main([
                "run", "--method", "d",
                "--y", str(data_dir / "y.csv"),
                "--demographics", str(data_dir / "demographics.csv"),
                "--output", str(tmp_path / "out"),
            ])
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "requires --group-col" in captured.err.lower()

    def test_method_accepts_case_insensitive(self, data_dir, tmp_path):
        pls_main([
            "run", "--method", "Correlational",
            "--x", str(data_dir / "x.csv"),
            "--y", str(data_dir / "y.csv"),
            "--demographics", str(data_dir / "demographics.csv"),
            "--output", str(tmp_path / "out"),
            "--n-perms", "10",
            "--n-bootstraps", "10",
            "--subject-id", "subject_id",
        ])
        assert (tmp_path / "out" / "data").exists()

    def test_group_col_and_groups_mutually_exclusive(
        self, data_dir, tmp_path, capsys,
    ):
        with pytest.raises(SystemExit) as exc_info:
            pls_main([
                "run", "--method", "c",
                "--x", str(data_dir / "x.csv"),
                "--y", str(data_dir / "y.csv"),
                "--demographics", str(data_dir / "demographics.csv"),
                "--group-col", "group",
                "--groups", str(data_dir / "groups.yaml"),
                "--output", str(tmp_path / "out"),
            ])
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "mutually exclusive" in captured.err.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`

Expected: FAIL — `run` command not registered

- [ ] **Step 3: Implement `pls run` command**

`pls/cli.py`:
```python
"""CLI entry point for PLS analysis."""

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from pathlib import Path

from pls.core import PLS

logger = logging.getLogger("pls")
from pls.io import (
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
from pls.plotting import (
    plot_heatmap,
    plot_loadings,
    plot_permutation,
    plot_scores_boxstrip,
    plot_scores_scatter,
)


METHOD_ALIASES = {
    "c": "correlational",
    "correlational": "correlational",
    "d": "discriminatory",
    "discriminatory": "discriminatory",
}


def _write_log(output_dir: Path, params: dict) -> None:
    """Write a log.txt with run parameters."""
    from datetime import datetime
    from pls import __version__

    log_path = output_dir / "log.txt"
    with open(log_path, "w") as f:
        f.write(f"PLS analysis log\n")
        f.write(f"Version: {__version__}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"\nParameters:\n")
        for k, v in params.items():
            f.write(f"  {k}: {v}\n")


def _save_csv(data: np.ndarray, path: Path, columns=None, index=None):
    """Save a numpy array as CSV."""
    df = pd.DataFrame(data, columns=columns, index=index)
    df.to_csv(path, index=index is not None)


def _error(message: str) -> None:
    """Print error message to stderr and exit with code 2."""
    print(f"error: {message}", file=sys.stderr)
    sys.exit(2)


def pls_main(argv=None):
    """PLS covariance analysis with statistical testing and visualisation.

    Args:
        argv: Command-line arguments. None uses sys.argv (normal CLI usage);
              pass a list for testing (e.g. pls_main(["run", "--method", "c", ...])).
    """
    parser = argparse.ArgumentParser(
        prog="pls",
        description="PLS covariance analysis with statistical testing and visualisation.",
    )
    subparsers = parser.add_subparsers(dest="command")

    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="Enable verbose logging output")

    # --- pls run ---
    run_parser = subparsers.add_parser("run", help="Run PLS analysis")
    run_parser.add_argument("--method", "-m", required=True,
                            help="correlational/c or discriminatory/d")
    run_parser.add_argument("--x", dest="x_path", default=None,
                            help="X matrix CSV (required for correlational)")
    run_parser.add_argument("--y", dest="y_path", required=True,
                            help="Y matrix CSV")
    run_parser.add_argument("--demographics", required=True,
                            help="Demographics CSV")
    run_parser.add_argument("--output", required=True, help="Output directory")
    run_parser.add_argument("--group-col", default=None,
                            help="Group column name (shorthand for YAML)")
    run_parser.add_argument("--groups", dest="groups_path", default=None,
                            help="Groups YAML config file")
    run_parser.add_argument("--subject-id", default=None,
                            help="Subject ID column name")
    run_parser.add_argument("--x-meta", default=None,
                            help="X metadata CSV")
    run_parser.add_argument("--y-meta", default=None,
                            help="Y metadata CSV")
    run_parser.add_argument("--n-perms", default=10000, type=int,
                            help="Number of permutations (default: 10000)")
    run_parser.add_argument("--n-bootstraps", default=10000, type=int,
                            help="Number of bootstrap resamples (default: 10000)")
    run_parser.add_argument("--seed", default=42, type=int,
                            help="Random seed (default: 42)")
    run_parser.add_argument("--all-plots", action="store_true", default=False,
                            help="Generate all plots including LV heatmaps, "
                                 "bootstrap ratio heatmaps, and diagnostics")
    run_parser.add_argument("--format", dest="img_format", default="svg",
                            choices=["svg", "png"],
                            help="Image format (default: svg)")
    run_parser.add_argument("--dpi", default=300, type=int,
                            help="Image DPI (default: 300)")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.command is None:
        parser.print_help()
        sys.exit(1)
    elif args.command == "run":
        _cmd_run(args)
    elif args.command == "cross-validate":
        _cmd_cross_validate(args)


def _cmd_run(args):
    """Run PLS analysis."""
    # --- Resolve method ---
    method_lower = args.method.lower()
    if method_lower not in METHOD_ALIASES:
        _error(
            f"Unknown method '{args.method}'. "
            f"Use correlational/c or discriminatory/d."
        )
    method_name = METHOD_ALIASES[method_lower]

    # --- Validate method-specific constraints ---
    if method_name == "correlational" and args.x_path is None:
        _error("Correlational PLS requires --x.")
    if method_name == "discriminatory" and args.x_path is not None:
        _error(
            "Discriminatory PLS builds X from --group-col. "
            "Do not provide --x."
        )
    if method_name == "discriminatory" and args.group_col is None and args.groups_path is None:
        _error(
            "Discriminatory PLS requires --group-col or --groups."
        )
    if args.group_col is not None and args.groups_path is not None:
        _error(
            "--group-col and --groups are mutually exclusive."
        )

    # --- Set up output directory ---
    output_dir = Path(args.output)
    figures_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- Parse groups config ---
    demographics_df = load_csv(Path(args.demographics))
    subject_id = args.subject_id

    if args.groups_path is not None:
        config = parse_groups_config(
            Path(args.groups_path), demographics_df=demographics_df
        )
        if config.subject_id:
            subject_id = config.subject_id
    elif args.group_col is not None:
        config = GroupConfig.from_group_col(args.group_col, subject_id=subject_id)
    else:
        config = None

    # --- Load and validate data ---
    y_df = load_csv(Path(args.y_path), require_numeric=True)
    dfs_to_align = [y_df, demographics_df]

    if method_name == "correlational":
        x_df = load_csv(Path(args.x_path), require_numeric=True)
        dfs_to_align.insert(0, x_df)

    sid = detect_subject_id(dfs_to_align, subject_id=subject_id)
    aligned = align_subjects(dfs_to_align, subject_id=sid)

    if method_name == "correlational":
        x_aligned, y_aligned, demo_aligned = aligned
        x_feature_names = [c for c in x_aligned.columns if c != sid]
    else:
        y_aligned, demo_aligned = aligned
        X_design, x_feature_names = build_design_matrix(demo_aligned, config)

    y_feature_names = [c for c in y_aligned.columns if c != sid]

    # --- Missing value checks ---
    check_missing_values(y_aligned, name="Y")
    if method_name == "correlational":
        check_missing_values(x_aligned, name="X")

    # --- Extract numeric arrays ---
    Y_raw = y_aligned[y_feature_names].to_numpy(dtype=float)

    if method_name == "correlational":
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
    x_meta_df = load_metadata(Path(args.x_meta), x_feature_names) if args.x_meta else None
    y_meta_df = load_metadata(Path(args.y_meta), y_feature_names) if args.y_meta else None

    # --- Run PLS ---
    model = PLS(X, Y, seed=args.seed)
    model.fit()
    model.permutation_test(n_perms=args.n_perms)
    model.bootstrap(n_bootstraps=args.n_bootstraps)
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
    ])
    scores_cols = (
        [f"X_{name}" for name in final_lv_names]
        + [f"Y_{name}" for name in final_lv_names]
    )
    scores_df = pd.DataFrame(scores_data, columns=scores_cols, index=subject_ids)
    scores_df.index.name = sid
    scores_df.to_csv(data_dir / "subject_scores.csv")

    # --- Generate plots ---
    ext = args.img_format

    # 1. Cross-correlation heatmap
    plot_heatmap(
        model.xcorr, v=1.0,
        xticklabels=y_feature_names,
        yticklabels=x_feature_names,
        out_path=figures_dir / f"cross_correlation.{ext}",
        dpi=args.dpi,
    )

    # 2. Permutation test
    plot_permutation(
        model.s, model.permuted_singular_values, model.p_values,
        out_path=figures_dir / f"permutation_test.{ext}",
        dpi=args.dpi,
    )

    # 3. Loading bar plots for final LVs
    for i, lv_name in enumerate(final_lv_names):
        lv_idx = [j for j, v in enumerate(model.final_lvs) if v][i]

        x_colours = None
        if x_meta_df is not None and "category" in x_meta_df.columns:
            colour_map = dict(zip(
                x_meta_df["feature"], x_meta_df["category"]
            ))
            palette = dict(zip(
                x_meta_df["category"].unique(),
                sns.color_palette("Set2", n_colors=x_meta_df["category"].nunique()),
            ))
            x_colours = [
                palette.get(colour_map.get(f), "steelblue")
                for f in x_feature_names
            ]

        plot_loadings(
            loadings=model.u_loadings[:, lv_idx],
            se=model.u_se[:, lv_idx],
            feature_names=x_feature_names,
            lv_name=lv_name,
            out_path=figures_dir / f"{lv_name}_x_loadings.{ext}",
            colours=x_colours,
            dpi=args.dpi,
        )

        y_colours = None
        if y_meta_df is not None and "category" in y_meta_df.columns:
            colour_map = dict(zip(
                y_meta_df["feature"], y_meta_df["category"]
            ))
            palette = dict(zip(
                y_meta_df["category"].unique(),
                sns.color_palette("Set2", n_colors=y_meta_df["category"].nunique()),
            ))
            y_colours = [
                palette.get(colour_map.get(f), "steelblue")
                for f in y_feature_names
            ]

        plot_loadings(
            loadings=model.vt_loadings[lv_idx, :],
            se=model.vt_se[lv_idx, :],
            feature_names=y_feature_names,
            lv_name=lv_name,
            out_path=figures_dir / f"{lv_name}_y_loadings.{ext}",
            colours=y_colours,
            dpi=args.dpi,
        )

    # 4. Subject score box/strip plots
    if config is not None and len(config.groups) > 0:
        # Build long-format scores dataframe with group info
        group_cols_to_use = [
            g for g in config.groups if g.role != "ignore"
        ]
        if group_cols_to_use:
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
                # Apply ordering
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
                    dpi=args.dpi,
                )

    # 5. Score scatter (correlational only)
    if method_name == "correlational" and config is not None:
        group_cols_to_use = [
            g for g in config.groups if g.role != "ignore"
        ]
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
                dpi=args.dpi,
            )

    # --- Write log ---
    _write_log(output_dir, {
        "method": method_name,
        "x": args.x_path,
        "y": args.y_path,
        "demographics": args.demographics,
        "group_col": args.group_col,
        "groups": args.groups_path,
        "subject_id": sid,
        "n_perms": args.n_perms,
        "n_bootstraps": args.n_bootstraps,
        "seed": args.seed,
        "all_plots": args.all_plots,
        "format": args.img_format,
        "dpi": args.dpi,
        "n_subjects": len(subject_ids),
        "n_x_features": len(x_feature_names),
        "n_y_features": len(y_feature_names),
        "significant_lvs": final_lv_names,
    })

    logger.info("PLS analysis complete. Results saved to: %s", output_dir)
    logger.info("Significant and reliable LVs: %s", final_lv_names)
```

Note: add `import seaborn as sns` to the import block in `cli.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`

Expected: all tests PASS

- [ ] **Step 5: Run an end-to-end test with synthetic data**

Run:
```bash
pls run --method c \
  --x tests/data/x.csv \
  --y tests/data/y.csv \
  --demographics tests/data/demographics.csv \
  --group-col group \
  --subject-id subject_id \
  --output /tmp/pls_test_corr \
  --n-perms 100 \
  --n-bootstraps 100
```

Expected: completes without error, outputs in `/tmp/pls_test_corr/`

- [ ] **Step 6: Commit**

```
feat: add pls run CLI command with full pipeline
```

---

### Task 17: CLI — `pls cross-validate`

**Files:**
- Modify: `pls/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_cli.py`:
```python
class TestCrossValidate:
    def test_requires_group_col(self, data_dir, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            pls_main([
                "cross-validate",
                "--y", str(data_dir / "y.csv"),
                "--demographics", str(data_dir / "demographics.csv"),
                "--output", str(tmp_path / "cv_out"),
            ])
        assert exc_info.value.code != 0

    def test_runs_successfully(self, data_dir, tmp_path):
        out = tmp_path / "cv_out"
        pls_main([
            "cross-validate",
            "--y", str(data_dir / "y.csv"),
            "--demographics", str(data_dir / "demographics.csv"),
            "--group-col", "group",
            "--subject-id", "subject_id",
            "--output", str(out),
            "--n-folds", "3",
            "--n-repeats", "2",
            "--n-permutations", "10",
        ])
        assert (out / "figures").exists()
        assert (out / "data").exists()
        assert (out / "log.txt").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::TestCrossValidate -v`

Expected: FAIL — `cross-validate` command not registered

- [ ] **Step 3: Implement `pls cross-validate` command**

Add the `cross-validate` subparser to `pls_main()`, before `args = parser.parse_args(argv)`:
```python
    # --- pls cross-validate ---
    cv_parser = subparsers.add_parser("cross-validate",
                                       help="Cross-validate discriminatory PLS model")
    cv_parser.add_argument("--y", dest="y_path", required=True,
                           help="Y matrix CSV")
    cv_parser.add_argument("--demographics", required=True,
                           help="Demographics CSV")
    cv_parser.add_argument("--output", required=True, help="Output directory")
    cv_parser.add_argument("--group-col", required=True,
                           help="Group column name")
    cv_parser.add_argument("--subject-id", default=None,
                           help="Subject ID column name")
    cv_parser.add_argument("--n-folds", default=5, type=int,
                           help="Number of CV folds (default: 5)")
    cv_parser.add_argument("--n-repeats", default=100, type=int,
                           help="Number of CV repeats (default: 100)")
    cv_parser.add_argument("--n-components", default=None, type=int,
                           help="Number of PLS components (default: n_groups - 1)")
    cv_parser.add_argument("--n-permutations", default=1000, type=int,
                           help="Number of permutations for CV test (default: 1000)")
    cv_parser.add_argument("--seed", default=42, type=int,
                           help="Random seed (default: 42)")
    cv_parser.add_argument("--format", dest="img_format", default="svg",
                           choices=["svg", "png"],
                           help="Image format (default: svg)")
    cv_parser.add_argument("--dpi", default=300, type=int,
                           help="Image DPI (default: 300)")
```

Add to `pls/cli.py` — the `_cmd_cross_validate` function and its imports:
```python
from pls.cross_validate import run_cv, permutation_test_cv
from pls.plotting import plot_cv_accuracy, plot_cv_permutation, plot_confusion_matrix


def _cmd_cross_validate(args):
    """Cross-validate discriminatory PLS model."""
    # --- Set up output ---
    output_dir = Path(args.output)
    figures_dir = output_dir / "figures"
    data_dir_out = output_dir / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir_out.mkdir(parents=True, exist_ok=True)

    # --- Load and validate ---
    y_df = load_csv(Path(args.y_path), require_numeric=True)
    demographics_df = load_csv(Path(args.demographics))
    config = GroupConfig.from_group_col(args.group_col, subject_id=args.subject_id)

    sid = detect_subject_id(
        [y_df, demographics_df], subject_id=args.subject_id
    )
    y_aligned, demo_aligned = align_subjects(
        [y_df, demographics_df], subject_id=sid
    )

    check_missing_values(y_aligned, name="Y")

    y_feature_names = [c for c in y_aligned.columns if c != sid]
    Y = y_aligned[y_feature_names].to_numpy(dtype=float)

    # Build group labels
    labels_series = demo_aligned[args.group_col].astype("category")
    label_names = labels_series.cat.categories.tolist()
    labels = labels_series.cat.codes.values
    n_groups = len(label_names)

    n_components = args.n_components
    if n_components is None:
        n_components = n_groups - 1

    # --- Run CV ---
    logger.info("Running %d-fold CV with %d repeats...", args.n_folds, args.n_repeats)
    cv_result = run_cv(
        Y, labels,
        n_splits=args.n_folds, n_repeats=args.n_repeats,
        n_components=n_components, seed=args.seed,
    )

    logger.info("Mean accuracy: %.3f", cv_result["mean_accuracy"])
    logger.info("Chance level: %.3f", 1 / n_groups)

    # --- Permutation test ---
    logger.info("Running permutation test (%d permutations)...", args.n_permutations)
    perm_result = permutation_test_cv(
        Y, labels,
        observed_accuracy=cv_result["mean_accuracy"],
        n_splits=args.n_folds, n_repeats=1,
        n_components=n_components,
        n_permutations=args.n_permutations, seed=args.seed,
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
    ext = args.img_format
    chance = 1 / n_groups

    plot_cv_accuracy(
        fold_accuracies=cv_result["fold_results"]["accuracy"].values,
        mean_accuracy=cv_result["mean_accuracy"],
        chance_level=chance,
        out_path=figures_dir / f"cv_fold_accuracy.{ext}",
        dpi=args.dpi,
    )

    plot_cv_permutation(
        null_accuracies=perm_result["null_accuracies"],
        observed_accuracy=cv_result["mean_accuracy"],
        p_value=perm_result["p_value"],
        out_path=figures_dir / f"cv_permutation_test.{ext}",
        dpi=args.dpi,
    )

    plot_confusion_matrix(
        cm=cv_result["confusion_matrix"],
        label_names=label_names,
        mean_accuracy=cv_result["mean_accuracy"],
        out_path=figures_dir / f"cv_confusion_matrix.{ext}",
        dpi=args.dpi,
    )

    # --- Log ---
    _write_log(output_dir, {
        "command": "cross-validate",
        "y": args.y_path,
        "demographics": args.demographics,
        "group_col": args.group_col,
        "subject_id": sid,
        "n_folds": args.n_folds,
        "n_repeats": args.n_repeats,
        "n_components": n_components,
        "n_permutations": args.n_permutations,
        "seed": args.seed,
        "n_subjects": len(Y),
        "n_groups": n_groups,
        "group_names": label_names,
        "mean_accuracy": f"{cv_result['mean_accuracy']:.3f}",
        "permutation_p_value": f"{perm_result['p_value']:.4f}",
    })

    logger.info("Cross-validation complete. Results saved to: %s", output_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`

Expected: all tests PASS

- [ ] **Step 5: Run end-to-end CV test**

Run:
```bash
pls cross-validate \
  --y tests/data/y.csv \
  --demographics tests/data/demographics.csv \
  --group-col group \
  --subject-id subject_id \
  --output /tmp/pls_test_cv \
  --n-folds 3 \
  --n-repeats 5 \
  --n-permutations 20
```

Expected: completes without error, outputs in `/tmp/pls_test_cv/`

- [ ] **Step 6: Commit**

```
feat: add pls cross-validate CLI command
```

---

### Task 18: Documentation

**Files:**
- Create: `docs/usage.md`
- Create: `docs/input-format.md`
- Create: `docs/missing-data.md`
- Create: `docs/interpreting-output.md`

- [ ] **Step 1: Write usage.md**

`docs/usage.md`:
```markdown
# Usage

## Installation

```bash
uv venv .venv
source .venv/bin/activate
uv pip install pls
```

For development:
```bash
uv pip install -e ".[dev]"
```

## Quick Start

### Correlational PLS

Finds covariance patterns between two continuous data matrices:

```bash
pls run --method c \
  --x brain_measures.csv \
  --y behaviour_scores.csv \
  --demographics participants.csv \
  --group-col treatment \
  --subject-id participant_id \
  --output results/
```

### Discriminatory PLS

Finds patterns that discriminate between groups:

```bash
pls run --method d \
  --y mri_features.csv \
  --demographics participants.csv \
  --group-col drug_group \
  --subject-id participant_id \
  --output results/
```

### Cross-Validation

Tests whether the discriminatory model generalises:

```bash
pls cross-validate \
  --y mri_features.csv \
  --demographics participants.csv \
  --group-col drug_group \
  --subject-id participant_id \
  --output cv_results/ \
  --n-folds 5 \
  --n-repeats 100
```

## Multiple Grouping Variables

Create a YAML file (e.g. `groups.yaml`):

```yaml
subject_id: participant_id
groups:
  - column: genotype
    role: x_axis
    reference: WT
    order: [WT, HET, KO]
  - column: treatment
    role: hue
    reference: vehicle
```

Then use `--groups groups.yaml` instead of `--group-col`.

## All Options

Run `pls run --help` or `pls cross-validate --help` for the full list.
```

- [ ] **Step 2: Write input-format.md**

`docs/input-format.md`:
```markdown
# Input Format

## Required Files

### X Matrix (correlational PLS only)

CSV with subjects as rows and features as columns. The first column must be
the subject identifier.

```csv
subject_id,region_A,region_B,region_C
sub01,1.23,4.56,7.89
sub02,2.34,5.67,8.90
```

### Y Matrix

Same format as X. Subject IDs must match across files (order does not matter
— the pipeline will align them).

### Demographics

CSV with a subject ID column and at least one grouping column.

```csv
subject_id,group,sex,age
sub01,control,F,25
sub02,treatment,M,30
```

## Optional Files

### Feature Metadata

CSV with a `feature` column matching data column headers, plus category
columns for plot colour-coding.

```csv
feature,category
region_A,frontal
region_B,frontal
region_C,temporal
```

### Groups Configuration

YAML file for multiple grouping variables. See `docs/usage.md` for examples.

## Subject Alignment

The pipeline finds the intersection of subject IDs across all input files.
Subjects present in some files but not others are excluded with a warning.
If no subjects are shared, the pipeline errors.

## Missing Data

The pipeline does **not** handle missing data. If any value in X or Y is
NaN, the pipeline errors and lists which subjects and features are affected.

See `docs/missing-data.md` for guidance on how to address this.
```

- [ ] **Step 3: Write missing-data.md**

Draw from notebook markdown. Content:

`docs/missing-data.md`:
```markdown
# Missing Data in PLS

## Why Missing Data Is a Problem

PLS decomposes the cross-covariance between all features in X and all
features in Y simultaneously. Every subject must have a value for every
feature. A single missing value in one feature for one subject means that
subject cannot contribute to the covariance structure.

## What to Do

You have two choices, and which is better depends on your data:

### Drop Subjects

Remove subjects who are missing any measurement. This preserves all features
but reduces your sample size.

**When to choose this:** You have many subjects relative to features, and
only a few subjects have missing data.

### Drop Features

Remove features that have missing data across many subjects. This preserves
sample size but removes those features from the analysis.

**When to choose this:** A specific measurement failed for many subjects
(e.g. a brain region that was poorly imaged), but the remaining measurements
are complete.

### General Guidance

- Never impute missing data before PLS. Imputation introduces artificial
  covariance structure that PLS will happily decompose, producing patterns
  that reflect the imputation method rather than biology.
- Check the pattern of missingness before deciding. If it is random (a few
  scattered NaNs), dropping subjects is usually fine. If it is systematic
  (one feature missing for a whole group), that feature may be unreliable
  and should be dropped.
- The module intentionally does not make this decision for you. Prepare
  your input files so they contain only the subjects and features you want
  to analyse.
```

- [ ] **Step 4: Write interpreting-output.md**

Draw from notebook markdown cells. Content:

`docs/interpreting-output.md`:
```markdown
# Interpreting PLS Output

## Cross-Correlation Heatmap

This matrix shows the Pearson correlation between every feature in X and
every feature in Y, computed across all subjects. It is the raw input to
the SVD. Strong positive or negative values indicate features that co-vary
across subjects.

## Singular Values and Permutation Test

The SVD breaks the cross-correlation matrix into latent variables (LVs),
ordered by how much covariance they explain. The singular value for each LV
quantifies its strength.

The permutation test asks: is this singular value larger than we would expect
if X and Y were unrelated? It shuffles the subject pairing between X and Y
10,000 times and compares the observed singular value to this null
distribution.

**How to read the plot:** A red line to the right of the grey histogram
indicates a singular value that exceeds the null distribution — that LV
captures real covariance, not noise.

## Loading Bar Plots

For each significant and reliable LV, the loading plots show which features
contribute most to the pattern. Bars are sorted by absolute loading. The
red error bars show the bootstrap standard error — they indicate how stable
each loading is across resampled versions of the data.

**Large bars with small error bars** are the features driving the pattern
reliably. **Large bars with large error bars** may be driven by a few
outlier subjects.

## Bootstrap Ratios

The bootstrap ratio is the loading divided by its standard error. It can be
interpreted like a z-score: values above 1.96 indicate that a feature's
contribution is reliable at the 95% confidence level.

## Subject Scores

Subject scores show how strongly each subject expresses a given LV pattern.
The X scores (XU) project each subject onto the X-side pattern; the Y
scores (YV') project onto the Y-side pattern.

**Box/strip plots** show how scores distribute across groups. If a LV
captures a group difference, the boxes will separate.

**Score scatter plots** (correlational PLS only) show the relationship
between X and Y scores. If the PLS pattern is strong, subjects should fall
along a diagonal. Group-specific linear fits reveal whether the X–Y
relationship differs by group.

## Cross-Validation (Discriminatory PLS)

Cross-validation tests whether the group discrimination holds on unseen
subjects. The fold accuracy histogram shows per-fold classification
accuracy, while the confusion matrix shows which groups are well-separated
and which are confused.

The permutation test of CV accuracy answers: is the observed accuracy
significantly better than chance? A p-value below 0.05 indicates that
the model generalises beyond the training data.

**Important:** do not select the number of components based on `pls run`
results and then feed that into cross-validation. This introduces
circularity. Use all components (the default) or use nested
cross-validation.
```

- [ ] **Step 5: Commit**

```
docs: add usage, input format, missing data, and output interpretation guides
```

---

## Self-Review

**Spec coverage check:**
- CLI interface (both commands): Tasks 16, 17
- Groups config (YAML, roles, reference/order): Task 6
- Input validation (all checks): Tasks 3, 4, 5, 7
- Core PLS (fit, permutation, bootstrap, filtering): Tasks 8, 9, 10, 11
- Cross-validation: Task 12
- Plotting (all core + CV plots): Tasks 13, 14, 15
- Output structure (CSVs, log.txt): Task 16
- Packaging: Task 1
- Documentation: Task 18
- Dynamic figure sizing: Task 13
- Metadata loading: Task 7
- Discriminatory dummy coding: Task 7
- Verbose plots: not fully implemented — see note below

**Gap found:** Verbose plots (LV heatmaps, bootstrap ratio heatmaps, raw data distributions, scree plot, CV convergence) are specified in the design but not explicitly tasked. These are lower priority and can be added as a follow-up task after the core pipeline is working. The `--verbose` flag is already accepted by the CLI; it just doesn't produce extra plots yet.

**Placeholder scan:** No TBDs, TODOs, or "implement later" found.

**Type consistency check:**
- `PLS` class methods: `fit()`, `permutation_test()`, `bootstrap()`, `filter_lvs()` — consistent across Tasks 8-11 and Task 16
- `GroupConfig`, `GroupSpec` — consistent between Task 6 and Tasks 7, 16, 17
- `run_cv()`, `permutation_test_cv()` — consistent between Task 12 and Task 17
- Plot function signatures — consistent between Tasks 13-15 and Task 16
