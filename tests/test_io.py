import pandas as pd
import pytest
from pathlib import Path
from plsdo.io import load_csv, detect_subject_id, align_subjects


class TestLoadCsv:
    def test_loads_valid_csv(self, data_dir):
        df = load_csv(data_dir / "brain.csv")
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (12, 6)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_csv(tmp_path / "nonexistent.csv")

    def test_empty_file_raises(self, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_csv(empty)

    def test_no_numeric_columns_raises(self, tmp_path):
        text_only = tmp_path / "text.csv"
        text_only.write_text("name,colour\nalice,red\nbob,blue\n")
        with pytest.raises(ValueError, match="no numeric columns"):
            load_csv(text_only, require_numeric=True)

    def test_unreadable_file_raises(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_bytes(b"\x80\x81\x82\x83")
        with pytest.raises(ValueError, match="could not be read"):
            load_csv(bad)


class TestDetectSubjectId:
    def test_explicit_subject_id(self, x_df, y_df, demographics_df):
        sid = detect_subject_id(
            [x_df, y_df, demographics_df], subject_id="subject_id"
        )
        assert sid == "subject_id"

    def test_explicit_subject_id_missing_from_file(self, x_df, y_df):
        with pytest.raises(
            ValueError, match="not found"
        ):
            detect_subject_id([x_df, y_df], subject_id="nonexistent")

    def test_auto_detect(self, x_df, y_df, demographics_df):
        sid = detect_subject_id([x_df, y_df, demographics_df])
        assert sid == "subject_id"

    def test_auto_detect_no_shared_column(self):
        df1 = pd.DataFrame({"a": [1], "val": [2]})
        df2 = pd.DataFrame({"b": [1], "val2": [3]})
        with pytest.raises(ValueError, match="No shared column"):
            detect_subject_id([df1, df2])


class TestAlignSubjects:
    def test_aligned_subjects_same_order(self, x_df, y_df, demographics_df):
        aligned = align_subjects(
            [x_df, y_df, demographics_df], subject_id="subject_id"
        )
        for df in aligned:
            assert list(df["subject_id"]) == [
                f"s{i:02d}" for i in range(1, 13)
            ]

    def test_reorders_mismatched(self):
        df1 = pd.DataFrame({"id": ["a", "b", "c"], "v1": [1, 2, 3]})
        df2 = pd.DataFrame({"id": ["c", "a", "b"], "v2": [30, 10, 20]})
        aligned = align_subjects([df1, df2], subject_id="id")
        assert list(aligned[0]["id"]) == list(aligned[1]["id"])

    def test_drops_non_shared_with_warning(self, capfd):
        df1 = pd.DataFrame({"id": ["a", "b", "c"], "v1": [1, 2, 3]})
        df2 = pd.DataFrame({"id": ["b", "c", "d"], "v2": [20, 30, 40]})
        aligned = align_subjects([df1, df2], subject_id="id")
        assert len(aligned[0]) == 2  # only b, c
        captured = capfd.readouterr()
        assert "not present in all files" in captured.out

    def test_empty_intersection_raises(self):
        df1 = pd.DataFrame({"id": ["a", "b"], "v1": [1, 2]})
        df2 = pd.DataFrame({"id": ["c", "d"], "v2": [3, 4]})
        with pytest.raises(ValueError, match="No subjects shared"):
            align_subjects([df1, df2], subject_id="id")
