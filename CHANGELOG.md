# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Breaking:** the CLI now uses one subcommand per method.
  `plsdo run --method correlational` and `plsdo run --method discriminatory`
  are replaced by `plsdo correlational` and `plsdo discriminatory`, with
  `corr` and `discrim` as short aliases. Each subcommand exposes only its
  relevant flags — `--x` is required by `correlational` and absent from
  `discriminatory` — so invalid combinations are rejected structurally rather
  than at runtime.
- Flag and subcommand-name abbreviations are no longer accepted
  (`allow_abbrev=False`), so saved invocations cannot silently break when a
  future flag makes a prefix ambiguous.
- The version is single-sourced from `plsdo/__init__.py` via hatchling's
  dynamic version; `pyproject.toml` no longer carries a separate number.

### Added

- Test coverage reporting (`pytest-cov`) with a 95% floor, enforced in CI.
- A CI guard that fails if `CITATION.cff` and the package version drift apart.
- End-to-end tests covering the full pipeline and cross-validation output.
- This changelog and a `CONTRIBUTING.md`.

### Fixed

- `facet_rows`/`facet_cols` group roles were parsed and validated but never
  applied — a config requesting faceting silently produced an unfaceted plot.
  The score box/strip plots are now faceted: `facet_rows` adds grid rows and
  `facet_cols` puts the facet on the columns (moving the latent variables onto
  the rows). Setting both is rejected with a clear error.
- `plsdo.__version__` was hardcoded to `0.1.0` while the package was `0.1.1`;
  the version is now correct and single-sourced.

## [0.1.1] - 2026-05-22

### Fixed

- Box-and-strip plots assigned colours in the wrong order; switched to a
  `FacetGrid` + `boxplot` construction so box and strip colours match.

### Added

- Zenodo DOI in the README and `CITATION.cff`.

## [0.1.0] - 2026-05-21

Initial release: correlational and discriminatory PLS via a single SVD engine,
permutation testing, bootstrap resampling with Procrustes alignment, two-stage
latent-variable filtering, cross-validation, a command-line interface, and a
suite of diagnostic plots.

[Unreleased]: https://github.com/braincentrekcl/plsdo/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/braincentrekcl/plsdo/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/braincentrekcl/plsdo/releases/tag/v0.1.0
