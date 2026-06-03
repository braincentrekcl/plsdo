import re

import pytest
from plsdo.cli import pls_main


class TestRunValidation:
    def test_no_subcommand_prints_help_and_exits(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            pls_main([])
        assert exc_info.value.code != 0
        assert "usage: plsdo" in capsys.readouterr().out

    def test_correlational_without_x_errors(self, data_dir, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            pls_main(
                [
                    "correlational",
                    "--y",
                    str(data_dir / "behaviour.csv"),
                    "--demographics",
                    str(data_dir / "demographics.csv"),
                    "--output",
                    str(tmp_path / "out"),
                ]
            )
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        # --x is structurally required on the correlational parser.
        assert "--x" in captured.err.lower()

    def test_discriminatory_with_x_errors(self, data_dir, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            pls_main(
                [
                    "discriminatory",
                    "--x",
                    str(data_dir / "brain.csv"),
                    "--y",
                    str(data_dir / "behaviour.csv"),
                    "--demographics",
                    str(data_dir / "demographics.csv"),
                    "--group-col",
                    "group",
                    "--output",
                    str(tmp_path / "out"),
                ]
            )
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        # The discriminatory parser has no --x; argparse rejects it.
        assert "unrecognized arguments" in captured.err.lower()

    def test_discriminatory_without_group_col_errors(self, data_dir, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            pls_main(
                [
                    "discriminatory",
                    "--y",
                    str(data_dir / "behaviour.csv"),
                    "--demographics",
                    str(data_dir / "demographics.csv"),
                    "--output",
                    str(tmp_path / "out"),
                ]
            )
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "requires --group-col" in captured.err.lower()

    def test_corr_alias_runs(self, data_dir, tmp_path):
        pls_main(
            [
                "corr",
                "--x",
                str(data_dir / "brain.csv"),
                "--y",
                str(data_dir / "behaviour.csv"),
                "--demographics",
                str(data_dir / "demographics.csv"),
                "--output",
                str(tmp_path / "out"),
                "--n-perms",
                "10",
                "--n-bootstraps",
                "10",
                "--subject-id",
                "subject_id",
            ]
        )
        assert (tmp_path / "out" / "data").exists()

    def test_discrim_alias_runs(self, data_dir, tmp_path):
        out = tmp_path / "out_discrim"
        pls_main(
            [
                "discrim",
                "--y",
                str(data_dir / "behaviour.csv"),
                "--demographics",
                str(data_dir / "demographics.csv"),
                "--group-col",
                "group",
                "--subject-id",
                "subject_id",
                "--output",
                str(out),
                "--n-perms",
                "10",
                "--n-bootstraps",
                "10",
            ]
        )
        assert (out / "data").exists()

    def test_group_col_and_groups_mutually_exclusive(
        self,
        data_dir,
        tmp_path,
        capsys,
    ):
        with pytest.raises(SystemExit) as exc_info:
            pls_main(
                [
                    "correlational",
                    "--x",
                    str(data_dir / "brain.csv"),
                    "--y",
                    str(data_dir / "behaviour.csv"),
                    "--demographics",
                    str(data_dir / "demographics.csv"),
                    "--group-col",
                    "group",
                    "--groups",
                    str(data_dir / "groups.yaml"),
                    "--output",
                    str(tmp_path / "out"),
                ]
            )
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        # argparse enforces the mutual exclusion structurally.
        assert "not allowed with" in captured.err.lower()

    def test_abbreviated_flags_are_rejected(self, data_dir, tmp_path, capsys):
        # allow_abbrev=False: --n-perm must not silently resolve to --n-perms, so
        # saved scripts cannot break when a future flag makes a prefix ambiguous.
        # All required flags are supplied so the only error is the abbreviation.
        with pytest.raises(SystemExit) as exc_info:
            pls_main(
                [
                    "discriminatory",
                    "--y",
                    str(data_dir / "behaviour.csv"),
                    "--demographics",
                    str(data_dir / "demographics.csv"),
                    "--group-col",
                    "group",
                    "--output",
                    str(tmp_path / "out"),
                    "--n-perm",
                    "10",
                ]
            )
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "unrecognized arguments" in captured.err.lower()

    def test_all_plots_creates_verbose_figures(self, data_dir, tmp_path):
        out = tmp_path / "out_allplots"
        pls_main(
            [
                "discriminatory",
                "--y",
                str(data_dir / "behaviour.csv"),
                "--demographics",
                str(data_dir / "demographics.csv"),
                "--group-col",
                "group",
                "--subject-id",
                "subject_id",
                "--output",
                str(out),
                "--n-perms",
                "10",
                "--n-bootstraps",
                "10",
                "--all-plots",
            ]
        )
        figs = out / "figures"
        assert (figs / "scree.svg").exists()
        assert (figs / "LV1_heatmap.svg").exists()
        assert (figs / "Y_raw_distributions.svg").exists()


class TestLogContents:
    def test_log_records_metadata_paths_and_bsr_threshold(self, data_dir, tmp_path):
        out = tmp_path / "out_log"
        pls_main(
            [
                "correlational",
                "--x",
                str(data_dir / "brain.csv"),
                "--y",
                str(data_dir / "behaviour.csv"),
                "--demographics",
                str(data_dir / "demographics.csv"),
                "--y-meta",
                str(data_dir / "behaviour_meta.csv"),
                "--output",
                str(out),
                "--n-perms",
                "10",
                "--n-bootstraps",
                "10",
                "--subject-id",
                "subject_id",
                "--bsr-threshold",
                "1.5",
            ]
        )
        log = (out / "log.txt").read_text()
        assert "y_meta:" in log
        assert str(data_dir / "behaviour_meta.csv") in log
        assert "x_meta: None" in log
        assert "bsr_threshold: 1.5" in log

    def test_log_records_none_when_no_metadata(self, data_dir, tmp_path):
        out = tmp_path / "out_log_none"
        pls_main(
            [
                "correlational",
                "--x",
                str(data_dir / "brain.csv"),
                "--y",
                str(data_dir / "behaviour.csv"),
                "--demographics",
                str(data_dir / "demographics.csv"),
                "--output",
                str(out),
                "--n-perms",
                "10",
                "--n-bootstraps",
                "10",
                "--subject-id",
                "subject_id",
            ]
        )
        log = (out / "log.txt").read_text()
        assert "x_meta: None" in log
        assert "y_meta: None" in log
        assert "bsr_threshold: 1.96" in log


def test_version_flag(capsys):
    # Tests the CLI wiring, not the version *value* (which check_version.py owns):
    # --version must exit 0 and emit "plsdo <real version>", catching a hardcoded
    # or empty version string without going stale on a literal.
    from plsdo import __version__

    with pytest.raises(SystemExit) as exc_info:
        pls_main(["--version"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out.strip()
    # Reports the package version (catches a hardcoded/decoupled version)...
    assert out == f"plsdo {__version__}"
    # ...and that version is well-formed (guard on the CLI output, not the import).
    assert re.match(r"^plsdo \d+\.\d+", out)


class TestCrossValidate:
    def test_requires_group_col(self, data_dir, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            pls_main(
                [
                    "cross-validate",
                    "--y",
                    str(data_dir / "behaviour.csv"),
                    "--demographics",
                    str(data_dir / "demographics.csv"),
                    "--output",
                    str(tmp_path / "cv_out"),
                ]
            )
        assert exc_info.value.code != 0

    def test_runs_successfully(self, data_dir, tmp_path):
        out = tmp_path / "cv_out"
        pls_main(
            [
                "cross-validate",
                "--y",
                str(data_dir / "behaviour.csv"),
                "--demographics",
                str(data_dir / "demographics.csv"),
                "--group-col",
                "group",
                "--subject-id",
                "subject_id",
                "--output",
                str(out),
                "--n-folds",
                "3",
                "--n-repeats",
                "2",
                "--n-permutations",
                "10",
            ]
        )
        assert (out / "figures").exists()
        assert (out / "data").exists()
        assert (out / "log.txt").exists()

    def test_accepts_groups_yaml(self, data_dir, tmp_path):
        out = tmp_path / "cv_yaml"
        pls_main(
            [
                "cross-validate",
                "--y",
                str(data_dir / "behaviour.csv"),
                "--demographics",
                str(data_dir / "demographics.csv"),
                "--groups",
                str(data_dir / "groups.yaml"),
                "--output",
                str(out),
                "--n-folds",
                "3",
                "--n-repeats",
                "2",
                "--n-permutations",
                "10",
            ]
        )
        assert (out / "log.txt").exists()
        log = (out / "log.txt").read_text()
        assert "group_col: group" in log
        assert "groups:" in log

    def test_group_col_and_groups_mutually_exclusive(self, data_dir, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            pls_main(
                [
                    "cross-validate",
                    "--y",
                    str(data_dir / "behaviour.csv"),
                    "--demographics",
                    str(data_dir / "demographics.csv"),
                    "--group-col",
                    "group",
                    "--groups",
                    str(data_dir / "groups.yaml"),
                    "--output",
                    str(tmp_path / "cv_both"),
                ]
            )
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        # argparse enforces the mutual exclusion structurally (matches the run side).
        assert "not allowed with" in captured.err.lower()

    def test_all_plots_creates_convergence_figure(self, data_dir, tmp_path):
        out = tmp_path / "cv_allplots"
        pls_main(
            [
                "cross-validate",
                "--y",
                str(data_dir / "behaviour.csv"),
                "--demographics",
                str(data_dir / "demographics.csv"),
                "--group-col",
                "group",
                "--subject-id",
                "subject_id",
                "--output",
                str(out),
                "--n-folds",
                "3",
                "--n-repeats",
                "5",
                "--n-permutations",
                "10",
                "--all-plots",
            ]
        )
        assert (out / "figures" / "cv_convergence.svg").exists()
