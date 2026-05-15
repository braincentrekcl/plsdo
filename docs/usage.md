# Usage

## Installation

```bash
git clone https://github.com/braincentrekcl/plsdo.git
cd plsdo
uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

For discriminatory PLS with cross-validation (requires scikit-learn):
```bash
uv pip install -e ".[cv]"
```

For development:
```bash
uv pip install -e ".[dev]"
```

## Quick Start

### Correlational PLS

Finds covariance patterns between two continuous data matrices:

```bash
plsdo run --method c \
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
plsdo run --method d \
  --y mri_features.csv \
  --demographics participants.csv \
  --group-col drug_group \
  --subject-id participant_id \
  --output results/
```

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

## All Options

Run `plsdo run --help` or `plsdo cross-validate --help` for the full list.
