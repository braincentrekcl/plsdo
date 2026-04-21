import pytest
from plsdo.cli import pls_main


class TestRunValidation:
    def test_method_required(self, data_dir):
        with pytest.raises(SystemExit) as exc_info:
            pls_main([
                "run",
                "--y", str(data_dir / "behaviour.csv"),
                "--demographics", str(data_dir / "demographics.csv"),
                "--output", "/tmp/pls_test",
            ])
        assert exc_info.value.code != 0

    def test_method_c_without_x_errors(self, data_dir, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            pls_main([
                "run", "--method", "c",
                "--y", str(data_dir / "behaviour.csv"),
                "--demographics", str(data_dir / "demographics.csv"),
                "--output", str(tmp_path / "out"),
            ])
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "correlational pls requires --x" in captured.err.lower()

    def test_method_d_with_x_errors(self, data_dir, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            pls_main([
                "run", "--method", "d",
                "--x", str(data_dir / "brain.csv"),
                "--y", str(data_dir / "behaviour.csv"),
                "--demographics", str(data_dir / "demographics.csv"),
                "--group-col", "group",
                "--output", str(tmp_path / "out"),
            ])
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "do not provide --x" in captured.err.lower()

    def test_method_d_without_group_col_errors(self, data_dir, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            pls_main([
                "run", "--method", "d",
                "--y", str(data_dir / "behaviour.csv"),
                "--demographics", str(data_dir / "demographics.csv"),
                "--output", str(tmp_path / "out"),
            ])
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "requires --group-col" in captured.err.lower()

    def test_method_accepts_case_insensitive(self, data_dir, tmp_path):
        pls_main([
            "run", "--method", "Correlational",
            "--x", str(data_dir / "brain.csv"),
            "--y", str(data_dir / "behaviour.csv"),
            "--demographics", str(data_dir / "demographics.csv"),
            "--output", str(tmp_path / "out"),
            "--n-perms", "10",
            "--n-bootstraps", "10",
            "--subject-id", "subject_id",
        ])
        assert (tmp_path / "out" / "data").exists()

    def test_group_col_and_groups_mutually_exclusive(
        self, data_dir, tmp_path, capsys,
    ):
        with pytest.raises(SystemExit) as exc_info:
            pls_main([
                "run", "--method", "c",
                "--x", str(data_dir / "brain.csv"),
                "--y", str(data_dir / "behaviour.csv"),
                "--demographics", str(data_dir / "demographics.csv"),
                "--group-col", "group",
                "--groups", str(data_dir / "groups.yaml"),
                "--output", str(tmp_path / "out"),
            ])
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "mutually exclusive" in captured.err.lower()


class TestCrossValidate:
    def test_requires_group_col(self, data_dir, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            pls_main([
                "cross-validate",
                "--y", str(data_dir / "behaviour.csv"),
                "--demographics", str(data_dir / "demographics.csv"),
                "--output", str(tmp_path / "cv_out"),
            ])
        assert exc_info.value.code != 0

    def test_runs_successfully(self, data_dir, tmp_path):
        out = tmp_path / "cv_out"
        pls_main([
            "cross-validate",
            "--y", str(data_dir / "behaviour.csv"),
            "--demographics", str(data_dir / "demographics.csv"),
            "--group-col", "group",
            "--subject-id", "subject_id",
            "--output", str(out),
            "--n-folds", "3",
            "--n-repeats", "2",
            "--n-permutations", "10",
        ])
        assert (out / "figures").exists()
        assert (out / "data").exists()
        assert (out / "log.txt").exists()
