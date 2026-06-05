# Contributing to plsdo

Thanks for your interest in improving `plsdo`. This guide covers the
development setup and the conventions the project follows.

## Development setup

`plsdo` uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management.

```bash
# Create and activate a virtual environment
uv venv .venv && source .venv/bin/activate

# Install the package with development dependencies
uv pip install -e ".[dev]"
```

The `dev` extra includes the test runner, coverage, the linter, and
`pre-commit`. Cross-validation additionally requires the `cv` extra
(`uv pip install -e ".[dev,cv]"`).

Install the git hooks once so ruff runs automatically on every commit:

```bash
pre-commit install
```

## Running tests

```bash
# Run the whole suite
.venv/bin/pytest tests/

# Run a single file or test
.venv/bin/pytest tests/test_core.py
.venv/bin/pytest tests/test_core.py::TestBootstrap::test_seed_reproducibility

# Run with coverage (a 95% floor is enforced in CI)
.venv/bin/pytest tests/ --cov=plsdo
```

Each new function should have at least one positive test (correct input →
correct output) and one negative test (invalid input → informative error).

## Linting and formatting

The project uses [ruff](https://docs.astral.sh/ruff/) for both:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format .
```

CI runs the linter, the formatter check, and the test suite on Python
3.10–3.12. Please make sure all three pass before opening a pull request.

## Conventions

- **British English** in all prose: documentation, commit messages,
  user-facing strings, and comments.
- **Type hints** on all function signatures.
- **Commit messages** use a conventional prefix and a single-line subject:
  `feat`, `fix`, `enh`, `ref`, `test`, `docs`, `chore`. Make small, logically
  coherent commits rather than one large batch.
- **No data in the package.** Test data lives in `tests/data/` and is synthetic
  and small.
- Note any user-facing change in `CHANGELOG.md` under the `Unreleased` heading.

## Scope

`plsdo` deliberately stays lean: it implements PLS and its associated
reliability machinery (permutation testing, bootstrap ratios), not a general
statistics toolkit. New dependencies and new inference procedures beyond PLS
itself need a clear justification. When in doubt, open an issue to discuss
before implementing.

## Reporting issues

Use the issue templates under
[`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE) for bug reports, feature
requests, and documentation issues.
