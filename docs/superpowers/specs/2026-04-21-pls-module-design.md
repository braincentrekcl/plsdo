# PLS Module Design Spec

A Python CLI for Partial Least Squares covariance analysis with built-in
statistical testing and visualization.

## Motivation

Two Jupyter notebooks implement correlational and discriminatory PLS for
neuroscience data. The methods are valid but tightly coupled to specific
datasets. This module extracts the general-purpose PLS pipeline into an
installable CLI that works on any pair of data matrices, produces
publication-quality plots, and keeps results fully transparent.

## Core Principle

Both correlational PLS (two continuous matrices) and discriminatory PLS
(design matrix vs continuous matrix) are the same operation: SVD of the
cross-covariance matrix. One SVD engine handles both cases. The user does not
need to choose a variant -- what they pass as X determines the behavior.

## CLI Interface

Two subcommands:

### `plsdo run` -- inference

```
# Correlational PLS (two continuous matrices):
plsdo run --method c \
  --x X.csv \
  --y Y.csv \
  --demographics demo.csv \
  --group-col Drug \
  --output results/ \
  --seed 42

# Discriminatory PLS (design matrix built from group column):
plsdo run --method d \
  --y Y.csv \
  --demographics demo.csv \
  --group-col Drug \
  --output results/ \
  --seed 42

# With optional metadata and multi-group config:
plsdo run --method correlational \
  --x X.csv \
  --y Y.csv \
  --demographics demo.csv \
  --x-meta xmeta.csv \
  --y-meta ymeta.csv \
  --groups groups.yaml \
  --output results/ \
  --verbose \
  --seed 42
```

Parameters:
- `--method`: required. Accepts `correlational`/`c` or `discriminatory`/`d`
  (case-insensitive).
- `--y`, `--demographics`: required CSV paths
- `--x`: required for correlational, forbidden for discriminatory
- `--output`: required output directory
- `--group-col`: single grouping column from demographics. Required for
  discriminatory (used to build the dummy-coded X matrix). Optional for
  correlational (used for plot grouping only).
- `--groups`: YAML config for multiple grouping columns (see below).
  Mutually exclusive with `--group-col`.
- `--subject-id`: name of the subject ID column shared across input files.
  Optional if using `--groups` YAML (which has its own `subject_id` field).
  If omitted entirely, auto-detected from shared column names with a warning.
- `--x-meta`, `--y-meta`: optional metadata CSVs for feature categorization
- `--n-perms`: number of permutations (default 10000)
- `--n-bootstraps`: number of bootstrap resamples (default 10000)
- `--seed`: random seed (default 42)
- `--verbose`: produce all plots, not just the core set
- `--format`: output image format, svg or png (default svg)
- `--dpi`: output resolution (default 300)

Method-specific behavior:
- `correlational`: z-scores both X and Y, variance checks on both
- `discriminatory`: builds X as dummy codes from `--group-col` in
  demographics, z-scores Y only. After dummy coding, checks for zero-variance
  columns in X (a group level with no subjects) and errors if found

CLI validation:
- `--method c` without `--x`: error ("correlational PLS requires --x")
- `--method d` with `--x`: error ("discriminatory PLS builds X from
  --group-col, do not provide --x")
- `--method d` without `--group-col`: error ("discriminatory PLS requires
  --group-col")

### `plsdo cross-validate` -- internal validation

```
plsdo cross-validate \
  --y Y.csv \
  --demographics demo.csv \
  --group-col Drug \
  --output results/ \
  --n-folds 5 \
  --n-repeats 100 \
  --n-permutations 1000 \
  --seed 42
```

Cross-validation is always discriminatory (predicting group membership from
continuous data). It does not require `--method` or `--x`. It builds the
design matrix from `--group-col` internally.

Parameters:
- `--y`, `--demographics`, `--output`, `--group-col`: required
- `--seed`: random seed (default 42)
- `--n-folds`: number of CV folds (default 5)
- `--n-repeats`: number of CV repeats (default 100)
- `--n-components`: PLS components for CV (default: n_groups - 1)
- `--n-permutations`: permutations for CV significance test (default 1000)

CV uses all components by default. Users must not set `--n-components` based
on results from `plsdo run` -- this introduces circularity. The documentation
must state this explicitly.

Internally, the CV uses Y as the predictor (what we observe for a new subject)
and the dummy-coded group matrix as the target (what we want to predict). It
uses sklearn's PLSRegression, which is distinct from the SVD-based core PLS.
This is intentional: prediction requires sklearn's `predict()` method.

## Groups Configuration

For multiple grouping variables, a YAML file defines column roles:

```yaml
subject_id: shortID
groups:
  - column: Genotype
    role: x_axis
    reference: WT
    order: [WT, NRXN, NLGN]
  - column: Drug
    role: hue
    reference: saline
    order_by: DrugNo
  - column: Sex
    role: facet_cols
    facet_col_wrap: 2
  - column: CageNumber
    role: ignore
```

The `subject_id` field specifies which column contains subject identifiers
across all input files.

Roles map to plot aesthetics:
- `x_axis`: categorical axis on box/strip plots
- `hue`: color encoding
- `facet_rows`: separate subplot rows
- `facet_cols`: separate subplot columns
- `ignore`: explicitly excluded from analysis and plotting

Optional per-column fields:
- `reference`: the baseline/reference level for this factor. Controls which
  level is first in plots and (for discriminatory PLS) which level is the
  baseline for dummy coding.
- `order`: explicit list of level names defining the display order.
- `order_by`: name of another column in demographics whose values define the
  sort order (e.g. `DrugNo` for `Drug`). Mutually exclusive with `order`.
- `facet_col_wrap`: number of columns before wrapping to the next row. Only
  applies to `facet_cols` role. If omitted, auto-picked for aesthetics.

YAML validation:
- A YAML column that does not exist in the demographics file: error
- Demographics columns not listed in the YAML (and not the subject ID
  column) are ignored, with a warning listing which columns were dropped:

```
INFO: Using 'shortID' as subject ID column
WARNING: Ignoring demographics columns not in groups config: AnimalID, GenotypeNo
```

If `--group-col` is used instead (single column), it defaults to `x_axis`.
A `--subject-id` CLI flag specifies the subject ID column for the simple
case. If `--subject-id` is not provided and no YAML is used, the pipeline
auto-detects by finding the first column name shared across all input files,
and warns about what was chosen.

### Interaction with `--method`

**Discriminatory PLS:** all group columns serve double duty. They are
dummy-coded additively and concatenated to form the X design matrix, and their
roles control plot aesthetics. Additive coding means each factor gets its own
set of dummy columns (e.g. Genotype = 3 columns, Drug = 2 columns, X = 5
columns total). This preserves main effects -- the SVD can find latent
variables driven by one factor, the other, or both. Interaction effects emerge
from the data without explicit interaction coding.

**Correlational PLS:** group columns are only used for plot aesthetics. X
comes from the user-provided CSV.

## Input Specification

### Required files

| File | Format | Requirements |
|------|--------|--------------|
| X matrix | CSV | Subjects as rows, features as columns. First column is subject ID. |
| Y matrix | CSV | Same format. Same subject IDs as X. |
| Demographics | CSV | Must share a subject ID column with X and Y. At least one grouping column. |

### Optional files

| File | Format | Purpose |
|------|--------|---------|
| X metadata | CSV | Must have a `feature` column whose values match X's column headers. Additional columns define categories (e.g. `category`, `region_group`). Used to color-code plots. Does not filter features. |
| Y metadata | CSV | Same structure, with `feature` values matching Y's column headers. |

Metadata does not control which features are included in the analysis -- all
features in X and Y are always used. Metadata only describes features for
plot aesthetics (color-coding by category). Validation:
- Feature in data but not in metadata: warn that it won't be color-coded
- Feature in metadata but not in data: error (likely a typo or stale metadata)

### Validation pipeline (in order)

1. Load all CSVs. Error if any file is missing, unreadable, empty, not CSV,
   or has no numeric columns where expected.
2. Identify subject ID column: use `subject_id` from YAML, `--subject-id`
   from CLI, or auto-detect (first column name shared across all input
   files). If auto-detected, warn with the chosen column name. Error if no
   shared column is found.
3. Check for missing values in X and Y. Hard error listing which
   subjects/features have NaNs. The module does not impute or drop data.
4. Align subjects: find the intersection of IDs across all files, reorder to
   match. Warn if any subjects are dropped (present in some files but not
   others).
5. Error if the intersection is empty.
6. Check feature variance (before z-scoring). Zero-variance features are a
   hard error (z-scoring would produce NaN). Near-zero variance features
   produce a warning listing the affected features and the percentage of
   identical values.
7. Z-score X and Y within features (column-wise).

## Package Structure

```
plsdo/
  __init__.py
  cli.py              -- argparse-based CLI entry point
  core.py             -- PLS class: SVD, permutation, bootstrap, scoring
  cross_validate.py   -- CV logic, uses sklearn PLSRegression
  io.py               -- load CSVs, validate inputs, align subjects
  plotting.py         -- all plot functions, stateless
pyproject.toml
uv.lock
tests/
  test_io.py
  test_core.py
  test_cross_validate.py
  test_plotting.py
  data/               -- small synthetic test datasets
docs/
  usage.md
  input-format.md
  missing-data.md
  interpreting-output.md
```

## Core Computation (`core.py`)

A single `PLS` class handles both correlational and discriminatory PLS.

### `PLS.__init__(X, Y, seed=None)`

Stores the input matrices and seed. No computation yet.

### `PLS.fit()`

1. Compute cross-correlation: `R = X'Y / (n-1)` (X and Y already z-scored)
2. Call `self._decompose()` (see below)
3. Compute loadings: `U_load = U @ diag(S)`, `Vt_load = diag(S) @ Vt`
4. Compute subject scores: `X_scores = X @ U`, `Y_scores = Y @ Vt.T`

### `PLS._decompose()`

```python
def _decompose(self):
    self.u, self.s, self.vt = np.linalg.svd(self.xcorr, full_matrices=False)
```

This method is separated to allow future subclassing for sparse PLS. A
`SparsePLS(PLS)` subclass would override only `_decompose()` and inherit all
other methods (permutation, bootstrap, scoring, etc.).

### `PLS.permutation_test(n_perms=10000)`

1. Permute rows of X, recompute SVD, collect singular values.
2. p-value = proportion of permuted singular values >= observed.
3. Stores: `p_values`, `permuted_singular_values`, `significant_lvs` (boolean
   mask at p < 0.05).

Requires `.fit()` to have been called. Raises error otherwise.

### `PLS.bootstrap(n_bootstraps=10000)`

1. Resample subjects with replacement.
2. Recompute SVD on resampled z-scored data.
3. Procrustes rotation on Vt to align with reference, then apply to scaled
   loadings.
4. Sign correction via dot product with reference loadings.
5. Compute standard errors across bootstrap samples.
6. Bootstrap ratios = loadings / SE.
7. Stores: `u_bootstrap_ratios`, `vt_bootstrap_ratios`, `u_se`, `vt_se`.

Requires `.fit()` to have been called. Raises error otherwise.

### Latent variable filtering

After permutation and bootstrap, LVs are filtered to keep only those that:
1. Are significant (permutation p < 0.05)
2. Have at least one feature with |bootstrap ratio| > 1.96 on both the X and
   Y sides

This is stored as a `final_lvs` boolean mask. The filtering selects latent
variables, not features. All features remain in the model.

### Random state

A single seed at the class level. Permutation and bootstrap draw from
sequential random state (one seed reproduces the full analysis).

## Cross-Validation (`cross_validate.py`)

Repeated Stratified K-Fold CV for discriminatory PLS.

For each fold:
1. Split into train/test.
2. Standardize X using training-set statistics only (no leakage).
3. Fit sklearn PLSRegression on training data.
4. Predict group-dummy scores for test subjects.
5. Classify by argmax of predicted dummy vector.

Uses all components (n_groups - 1) by default. Does not use permutation or
bootstrap results from `plsdo run` -- this avoids circularity.

Permutation test of CV accuracy: shuffle group labels, repeat full CV, build
null distribution of accuracy.

## Plotting (`plotting.py`)

All plot functions are stateless: they take data arrays, optional metadata,
and an output path. They are not methods on the PLS class.

### Dynamic sizing

Figure dimensions scale with data shape:

```python
def figure_size(n_rows, n_cols, cell_size=0.5, min_dim=4, max_dim=40):
    width = max(min_dim, min(max_dim, n_cols * cell_size + 2))
    height = max(min_dim, min(max_dim, n_rows * cell_size + 2))
    return (width, height)
```

Heatmap annotations are suppressed when either axis exceeds 30 features.

### Core plots (`plsdo run`)

1. **Cross-correlation heatmap**: R matrix, X features as rows, Y as columns.
   Color-coded side bars if metadata provided.
2. **Permutation test histograms**: one subplot per LV, grey null
   distribution, red line for observed singular value.
3. **Loading bar plots**: one figure per significant+reliable LV, separately
   for X and Y. Horizontal bars sorted by |loading|, SE error bars.
   Color-coded by category if metadata provided.
4. **Subject score box/strip plots**: one panel per significant+reliable LV.
   Grouping from `--group-col` or `--groups` config.
5. **Score scatter plots** (correlational only): XU vs YV' per
   significant+reliable LV. Points by group, linear fit lines per group.
   Not produced for discriminatory PLS since X is categorical.

Method-specific plot behavior:
- **Correlational**: all 5 core plots produced. X and Y loading bar plots
  both show continuous features.
- **Discriminatory**: X loading bar plots show group-level loadings. These
  are core for designs with many groups (8+) or multiple factors, but may be
  less informative with few groups. Always produced, but the user should
  interpret accordingly. Score scatter plots are not produced.

### Core plots (`plsdo cross-validate`)

1. **Fold accuracy histogram**: per-fold accuracies with mean and chance.
2. **Permutation null distribution**: observed accuracy vs null, with p-value.
3. **Confusion matrix heatmap**: normalized by true class.

### Verbose additions (`plsdo run`)

- Latent variable heatmaps (rank-1 R reconstruction per LV)
- Bootstrap ratio heatmaps for X and Y
- Raw data distribution plots (z-scored features by group)
- Singular value scree plot

### Verbose additions (`plsdo cross-validate`)

- Convergence plot (cumulative mean accuracy over repeats)
- Configuration sweep comparison

### Output format

SVG by default. PNG via `--format png`. DPI configurable (default 300).

## Output Structure

### `plsdo run`

```
{output}/
  figures/
    cross_correlation.svg
    permutation_test.svg
    LV1_x_loadings.svg
    LV1_y_loadings.svg
    LV1_scores_boxplot.svg
    LV1_scores_scatter.svg
    LV2_x_loadings.svg
    ...
  data/
    singular_values.csv
    p_values.csv
    x_loadings.csv
    y_loadings.csv
    x_bootstrap_ratios.csv
    y_bootstrap_ratios.csv
    subject_scores.csv
  log.txt
```

### `plsdo cross-validate`

```
{output}/
  figures/
    cv_fold_accuracy.svg
    cv_permutation_test.svg
    cv_confusion_matrix.svg
  data/
    cv_fold_results.csv
    cv_permutation_accuracies.csv
  log.txt
```

### `log.txt`

Records all parameters, file paths, seed, package version, and timestamp for
reproducibility.

## Testing

### `test_io.py`

- Unreadable files (missing, empty, not CSV, no numeric columns): error
- No shared column name across files and no `--subject-id` provided: error
- `--subject-id` or YAML `subject_id` specifies a column that doesn't exist
  in one or more files: error
- Missing values in X or Y: error with subject/feature names
- Mismatched subject IDs: reorder when possible, error when intersection empty
- Subjects present in some files but not others: reorder + warn
- Z-scoring: zero-mean, unit-variance per column
- Zero-variance features: hard error
- Near-zero variance features: warning with feature names and % identical
- YAML references a column not in demographics: error
- Demographics columns not in YAML: warning listing dropped columns
- Discriminatory dummy coding produces zero-variance column (empty group): error
- Feature in data but not in metadata: warning
- Feature in metadata but not in data: error

### `test_core.py`

- Known small matrix with hand-computed SVD: verify U, S, Vt, loadings,
  scores
- Seed reproducibility: same seed produces identical permutation p-values and
  bootstrap ratios
- Calling `.permutation_test()` or `.bootstrap()` before `.fit()`: error
- LV filtering: verify correct inclusion/exclusion based on permutation
  significance and bootstrap ratio thresholds on both X and Y sides

### `test_cross_validate.py`

- Perfect separation data: accuracy near 1.0
- Random data: accuracy near chance
- Seed reproducibility

### `test_plotting.py`

- Each plot function runs without error on synthetic data
- Dynamic sizing: call `figure_size()` with known inputs, assert output
  dimensions are within min/max bounds
- Annotation suppression: call heatmap function with >30 features, check the
  returned axes object has no text annotations
- Score scatter not produced when method is discriminatory

All tests use small synthetic datasets in `tests/data/`. No real subject data
in the repository.

## Packaging

### `pyproject.toml`

```toml
[project]
name = "plsdo"
version = "0.1.0"
description = "PLS covariance analysis with statistical testing and visualisation"
license = "BSD-3-Clause"
authors = [
    { name = "Eilidh MacNicol" },
]
maintainers = [
    { name = "Eilidh MacNicol" },
]
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
]

[project.scripts]
plsdo = "plsdo.cli:pls_main"

[project.urls]
Repository = "https://github.com/braincentrekcl/plsdo"
Issues = "https://github.com/braincentrekcl/plsdo/issues"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Installation

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

`uv.lock` pins exact dependency versions for reproducibility.

## Documentation

```
docs/
  usage.md               -- installation, basic CLI examples
  input-format.md        -- CSV specifications with examples
  missing-data.md        -- why missing data is a problem, features vs subjects tradeoff
  interpreting-output.md -- what each plot means, how to read bootstrap ratios
```

Markdown files for now. Can migrate to ReadTheDocs/Sphinx before public
release.

## Reference Implementation

The existing notebooks (`correlational_pls.ipynb`, `discriminatory_pls.ipynb`)
and cross-validation script (`claude_cross_validation.py`) are the reference
implementations for computational steps and plot styling. These files are NOT
distributed with the package -- they contain data and results linked to
specific publications.

During implementation:
- Computational steps should match these notebooks unless there is a specific
  reason to diverge (discuss with the user first).
- Plot styling should preserve the visual choices in the notebooks. Suggestions
  for changes are welcome but must be discussed before implementation.
- Documentation text should draw from the markdown cells in the notebooks,
  verified for technical correctness before committing.

## Future Work (Not Built Now)

- **Sparse PLS**: subclass overriding `_decompose()` with L1-penalized
  optimization. CLI flag `--method sparse`.
- **Held-out validation**: `--x-train`, `--y-train`, `--x-test`, `--y-test`
  for genuine external validation datasets.
- **Nested CV**: inner loop for component selection, outer loop for
  evaluation.
- **Python API**: the PLS class is already usable programmatically, but
  documentation and official support for the API come later.
- **ReadTheDocs/Sphinx**: full documentation site for public release.
