# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install for development (from repo root, with .venv active)
uv pip install -e ".[dev]"

# Run all tests
.venv/bin/pytest tests/

# Run a single test file
.venv/bin/pytest tests/test_core.py

# Run a single test by name
.venv/bin/pytest tests/test_core.py::TestBootstrap::test_seed_reproducibility
```

## Architecture

The package has a strict separation of concerns across five modules:

- **`io.py`** — everything that touches files or validates inputs: loading CSVs, detecting subject IDs, aligning subjects, checking missing values and variance, z-scoring, parsing YAML group configs, loading feature metadata, and building the dummy-coded design matrix.
- **`core.py`** — the `PLS` class. Stateful: takes z-scored arrays, runs `fit()` → `permutation_test()` → `bootstrap()` → `filter_lvs()` in sequence. Stores results as instance attributes.
- **`cross_validate.py`** — `run_cv()` and `permutation_test_cv()`. Uses `sklearn.PLSRegression` (not the SVD-based `PLS` class) because prediction requires `predict()`. Entirely independent of `core.py`.
- **`plotting.py`** — stateless functions. All take data arrays and an `out_path`, save the figure, return nothing. `meta_colours()` is here too (not in pipeline).
- **`pipeline.py`** — orchestration only. Calls `io` → `core` → `plotting` in sequence, writes CSVs and `log.txt`. No computation here.
- **`cli.py`** — argument parsing and validation only. Dispatches to `pipeline.run_pipeline()` or `pipeline.cross_validate_pipeline()`.

### Key design decisions

**One SVD engine for both PLS variants.** Correlational PLS z-scores both X and Y; discriminatory PLS uses a dummy-coded X (not z-scored) and z-scores Y only. The `PLS` class handles both — `pipeline.py` builds the right inputs before calling it.

**Bootstrap uses Procrustes + sign correction.** Each bootstrap SVD is aligned to the reference via `scipy.linalg.orthogonal_procrustes` on Vt, then signs are corrected by dot product with the reference Vt loadings. Both U and Vt loadings are aligned together.

**LV filtering is two-stage.** `filter_lvs()` keeps LVs that are (1) significant by permutation (p < 0.05) and (2) have at least one feature with |bootstrap ratio| > 1.96 on *both* the X and Y sides. Result is a boolean `final_lvs` mask.

**CV flips X and Y.** `cross_validate.py` uses Y (continuous data) as the predictor and dummy-coded groups as the target, so `pls.predict()` gives predicted group scores. This is the opposite convention from `plsdo run`.

## Design philosophy

Four principles, in priority order:

1. **Mathematical validity** — correctness is non-negotiable.
2. **Lightweight** — no unnecessary dependencies. Every dependency must earn its place.
3. **Scientific Python standards** — follow community conventions so the package is citable, installable, and maintainable.
4. **Glass-box and FAIR** — output everything needed to reproduce a result; keep the implementation transparent.

Practical consequences: prefer stdlib over third-party where reasonable (argparse over click, logging over print). Do not add inference or statistical tests beyond PLS itself — plotting scores by group factors is in scope; pairwise post-hoc tests are not. When in doubt, do less.

**Efficiency** is part of correctness. Prefer vectorised NumPy operations over Python loops wherever the maths permits — not for micro-optimisation, but because this code runs on HPCs and environmental cost is real. If a loop can be replaced by array operations without adding complexity or obscuring intent, replace it.

**Robustness** sits inside principle 1, not alongside it. Validate aggressively anywhere a silent failure could propagate — at file boundaries and wherever mathematical assumptions could be violated (zero variance before z-scoring, empty group levels before dummy coding, etc.). Fail loudly with informative errors. Trust internal transformations between already-validated states; defensive checks between modules add noise without catching anything real.

## Conventions

- British English in all prose: docs, commit messages, user-facing strings, comments.
- Commit messages use conventional prefixes: `feat`, `fix`, `enh`, `ref`, `test`, `docs`, `chore`. User commits with a GPG key — stage files and provide message text only, with attribution for claude, do not run `git commit`.
- `plsdo/` contains no data. Test data lives in `tests/data/` (synthetic, small).
- Reference notebooks (`.dev/correlational_pls.ipynb`, `.dev/discriminatory_pls.ipynb`, `.dev/claude_cross_validation.py`) are the source of truth for computational steps and plot styling. Deviations require discussion. These files are gitignored and live only in your local working copy.
- `.dev/superpowers/specs/` and `.dev/superpowers/plans/` contain the design spec and implementation plan. Consult them before making structural changes. These files are gitignored.

## Before public release / PyPI submission
- Claim the `plsdo` package name on PyPI before announcing the package publicly — squatting is a real risk once there is any visibility.
- Update `README.md` and `docs/usage.md` installation instructions from `git clone` to `pip install plsdo` once the package is published.
- Bump version to `1.0.0` and update the `Development Status` classifier to `4 - Beta` or `5 - Production/Stable` as appropriate.
