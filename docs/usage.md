# Usage

## Installation

Requires Python ≥ 3.10.

```bash
uv pip install plsdo
```

For discriminatory PLS with cross-validation (requires scikit-learn):
```bash
uv pip install "plsdo[cv]"
```

For development:
```bash
git clone https://github.com/braincentrekcl/plsdo.git
cd plsdo
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Quick Start

### Correlational PLS

Finds covariance patterns between two continuous data matrices:

```bash
plsdo correlational \
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
plsdo discriminatory \
  --y mri_features.csv \
  --demographics participants.csv \
  --group-col drug_group \
  --subject-id participant_id \
  --output results/
```

`corr` and `discrim` are accepted as short aliases for `correlational`
and `discriminatory` (e.g. `plsdo corr ...`), and `cv` for
`cross-validate`.

### Cross-Validation

> **Requires the `[cv]` extra.** Install with `uv pip install "plsdo[cv]"` before using this subcommand. Without it you will get `ModuleNotFoundError: No module named 'sklearn'`.

Tests whether the discriminatory model generalises:

```bash
plsdo cross-validate \
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
This works for `plsdo correlational`, `plsdo discriminatory`, and
`plsdo cross-validate`; for
cross-validation the column with `role: x_axis` is used as the
classification target.

How these columns are used depends on the method:

- **Discriminatory PLS** builds its model *from* these columns: every column
  with a role other than `ignore` is dummy-coded into the design matrix. The
  role also chooses the plot layout. Use `role: ignore` to keep a column out
  of the model.
- **Correlational PLS** takes X and Y as your own matrices, so grouping
  columns never enter the model — they are used only to colour and facet the
  score plots. Here `role: ignore` simply means the column is not used for
  display.

In both cases the role (`x_axis`, `hue`, `facet_rows`, `facet_cols`) chooses
how that factor is laid out in the score plots.

### Faceting the Score Plots

The subject-score box/strip plots can be split into a grid by a further
grouping variable using the `facet_rows` or `facet_cols` role:

```yaml
groups:
  - column: genotype
    role: x_axis
  - column: sex
    role: facet_rows
```

Each latent variable is always shown, so it occupies one axis of the grid
and a facet takes the other:

- `facet_rows` keeps the latent variables across the columns and splits the
  facet levels down the rows (the usual choice).
- `facet_cols` puts the facet levels across the columns and moves the latent
  variables onto the rows. Use it when you want the latent variables shown as
  rows.

Because a grid has only two axes — one for the latent variables and one for a
facet — you may set **either** `facet_rows` **or** `facet_cols`, but not both.
A config that sets both is rejected with an error when it is parsed (the
latent variables and two facets cannot share two axes).

With no facet, the latent variables stay on the columns; add
`facet_col_wrap: N` to a group to control how many latent-variable columns
appear before wrapping.

### Compound Subject IDs

If subjects are identified by more than one column (e.g. a subject
scanned across multiple runs), specify a list in the YAML:

```yaml
subject_id: [subject_id, run_id]
groups:
  - column: drug
    role: x_axis
```

The pipeline aligns on the compound key and writes a multi-level
index in the subject scores CSV.
Compound keys are only supported via YAML — `--subject-id` accepts
a single column name.

## Feature Metadata (Colour-Coding Loading Plots)

To colour-code features by category in the loading bar plots, pass a
metadata CSV via `--x-meta` and/or `--y-meta`.
The file must have at least two columns: `feature` and `category`.

| feature                  | category |
|--------------------------|----------|
| L Amygdala early CBF     | CBF      |
| L Amygdala hurst         | Hurst    |
| ...                      | ...      |

Whitespace after the comma in headers is tolerated.
Feature names in the metadata file must match the data exactly.
Features present in the data but missing from the metadata are uncoloured.

## Filtering Loading Plots by Bootstrap Ratio

By default, loading bar plots show only features whose `|bootstrap ratio|`
exceeds `1.96` (≈ 95% CI under the standard-normal approximation).
Override with `--bsr-threshold <float>` — e.g. `--bsr-threshold 2.58`
for a stricter cut, or `--bsr-threshold 0` to plot every feature.
The underlying `x_loadings.csv`, `y_loadings.csv`, `x_bootstrap_ratios.csv`,
and `y_bootstrap_ratios.csv` are not filtered.

## Verbose Plots and High-Dimensional Data

`--all-plots` generates additional diagnostic figures (scree plot,
rank-1 heatmaps, bootstrap ratio heatmaps, raw feature distributions).
When `--x-meta`/`--y-meta` are supplied, the rank-1 and bootstrap-ratio
heatmaps gain category colour bars alongside their rows (X features) and
columns (Y features), using the same metadata categories as the loading
plots.
When the feature count exceeds 100, heatmaps and distribution plots
become unreadable and extremely slow, so only the scree plot is
produced.
Override with `--verbose-feature-limit N` if you need the full set
at higher dimensions — but the figures will degrade in quality.

## All Options

Run `plsdo correlational --help`, `plsdo discriminatory --help`, or
`plsdo cross-validate --help` for the full list.
