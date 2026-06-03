"""Tests for the version-drift guard in scripts/check_version.py.

The script is not an importable package, so load it by path.
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_version.py"
_spec = importlib.util.spec_from_file_location("check_version", _SCRIPT)
check_version = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_version)


class TestExtract:
    def test_finds_version(self, tmp_path):
        f = tmp_path / "init.py"
        f.write_text('__version__ = "1.2.3"\n')
        assert check_version._extract(f, r'__version__\s*=\s*"([^"]+)"') == "1.2.3"

    def test_finds_multiline_anchored_version(self, tmp_path):
        # The ^version: anchor must not match cff-version: on an earlier line.
        f = tmp_path / "CITATION.cff"
        f.write_text('cff-version: 1.2.0\nversion: "0.4.0"\n')
        assert check_version._extract(f, r'^version:\s*"?([^"\n]+)"?') == "0.4.0"

    def test_missing_version_exits(self, tmp_path):
        f = tmp_path / "init.py"
        f.write_text("# no version here\n")
        with pytest.raises(SystemExit):
            check_version._extract(f, r'__version__\s*=\s*"([^"]+)"')


class TestCompare:
    def test_agree_returns_none(self):
        assert check_version.compare("0.1.1", "0.1.1") is None

    def test_mismatch_returns_message(self):
        msg = check_version.compare("0.1.1", "9.9.9")
        assert msg is not None
        assert "0.1.1" in msg and "9.9.9" in msg
