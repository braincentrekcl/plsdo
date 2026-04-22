# plsdo

Partial Least Squares (PLS) covariance analysis with permutation testing, bootstrap reliability, and publication-ready visualisation — from the command line.

(Pronounced: "please do")

`plsdo` was built out of necessity for project-specific neuroscience and neuroimaging pipelines, then generalised to handle flexible, diverse datasets beyond its origins. It implements two PLS variants used in neuroimaging and cognitive neuroscience research:

- **Correlational PLS** — finds latent variables that maximise covariance between two continuous data matrices (e.g. brain measures and behaviour scores).
- **Discriminatory PLS** — finds latent variables that maximise covariance between a continuous data matrix and a dummy-coded group matrix (i.e. group differences).

Statistical validity is built in: every analysis runs a permutation test on singular values and bootstraps loading stability. Only latent variables that pass both tests appear in the output.

---

## Installation

Requires Python ≥ 3.10.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install plsdo
```

For discriminatory PLS with cross-validation (requires scikit-learn):
```bash
uv pip install "plsdo[cv]"
```

Or with pip:
```bash
pip install plsdo
pip install "plsdo[cv]"   # for cross-validation
```

---

## Quick start

### Correlational PLS

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

```bash
plsdo run --method d \
  --y mri_features.csv \
  --demographics participants.csv \
  --group-col drug_group \
  --subject-id participant_id \
  --output results/
```

### Cross-validation (discriminatory only)

Requires `plsdo[cv]` — see Installation above.

```bash
plsdo cross-validate \
  --y mri_features.csv \
  --demographics participants.csv \
  --group-col drug_group \
  --subject-id participant_id \
  --output cv_results/
```

---

## Output

Each run writes to the output directory:

```
results/
  figures/     cross-correlation heatmap, permutation test, loading bar plots, score plots
  data/        singular values, p-values, loadings, bootstrap ratios, subject scores (CSV)
  log.txt      parameters and version stamp
```

---

## Documentation

| Page | Contents |
|------|----------|
| [Usage guide](docs/usage.md) | Full CLI options, multiple grouping variables, all flags |
| [Input format](docs/input-format.md) | How to structure X, Y, demographics, and metadata files |
| [Interpreting output](docs/interpreting-output.md) | What each plot and CSV means |
| [Missing data](docs/missing-data.md) | Why plsdo does not impute, and what to do instead |

---

## Contributing

Issues and pull requests are welcome. Please open an issue before starting significant work.

Contact: eilidh [dot] macnicol [at] kcl [dot] ac [dot] uk

---

## Licence

BSD 3-Clause. See [LICENSE](LICENSE).
