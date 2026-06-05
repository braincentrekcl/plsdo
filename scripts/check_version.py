#!/usr/bin/env python3
"""Fail if the package version and CITATION.cff version disagree.

The canonical version lives in ``plsdo/__init__.py`` (read dynamically by
hatchling). ``CITATION.cff`` carries an independent copy for citation
metadata. This guard keeps the two in step — run it in CI and before
cutting a release.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _extract(path: Path, pattern: str) -> str:
    text = path.read_text()
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        sys.exit(f"Could not find a version in {path}")
    return match.group(1)


def compare(package_version: str, citation_version: str) -> str | None:
    """Return an error message if the two versions disagree, else None."""
    if package_version != citation_version:
        return (
            f"Version mismatch: plsdo/__init__.py is {package_version!r} but "
            f"CITATION.cff is {citation_version!r}. Update both (and "
            f"CITATION.cff's date-released) before releasing."
        )
    return None


def main() -> None:
    package_version = _extract(
        ROOT / "plsdo" / "__init__.py", r'__version__\s*=\s*"([^"]+)"'
    )
    citation_version = _extract(
        ROOT / "CITATION.cff",
        r'^version:\s*"?([^"\n]+)"?',
    )

    error = compare(package_version, citation_version)
    if error:
        sys.exit(error)

    print(f"Versions agree: {package_version}")


if __name__ == "__main__":
    main()
