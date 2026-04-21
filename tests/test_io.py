import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import yaml
from plsdo.io import (
    load_csv, detect_subject_id, align_subjects,
    check_missing_values, check_variance, zscore_columns,
    parse_groups_config, GroupConfig,
)


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

    def test_drops_non_shared_with_warning(self, caplog):
        df1 = pd.DataFrame({"id": ["a", "b", "c"], "v1": [1, 2, 3]})
        df2 = pd.DataFrame({"id": ["b", "c", "d"], "v2": [20, 30, 40]})
        with caplog.at_level("WARNING", logger="plsdo"):
            aligned = align_subjects([df1, df2], subject_id="id")
        assert len(aligned[0]) == 2  # only b, c
        assert "not present in all files" in caplog.text

    def test_empty_intersection_raises(self):
        df1 = pd.DataFrame({"id": ["a", "b"], "v1": [1, 2]})
        df2 = pd.DataFrame({"id": ["c", "d"], "v2": [3, 4]})
        with pytest.raises(ValueError, match="No subjects shared"):
            align_subjects([df1, df2], subject_id="id")


class TestCheckMissingValues:
    def test_no_missing_passes(self):
        df = pd.DataFrame({"id": ["a", "b"], "v1": [1.0, 2.0], "v2": [3.0, 4.0]})
        check_missing_values(df, name="test")  # should not raise

    def test_missing_values_raises(self):
        df = pd.DataFrame({"id": ["a", "b"], "v1": [1.0, float("nan")], "v2": [3.0, 4.0]})
        with pytest.raises(ValueError, match="missing values"):
            check_missing_values(df, name="test")

    def test_reports_which_subjects_and_features(self):
        df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "v1": [1.0, float("nan"), 3.0],
            "v2": [float("nan"), 2.0, 4.0],
        })
        with pytest.raises(ValueError, match="v1.*v2"):
            check_missing_values(df, name="test")


class TestCheckVariance:
    def test_normal_variance_passes(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        check_variance(arr, feature_names=["a", "b"])  # should not raise

    def test_zero_variance_raises(self):
        arr = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
        with pytest.raises(ValueError, match="zero variance"):
            check_variance(arr, feature_names=["const", "varying"])

    def test_near_zero_variance_warns(self, caplog):
        # 9 out of 10 values are identical
        arr = np.array([[1.0, 5.0]] * 9 + [[2.0, 6.0]])
        with caplog.at_level("WARNING", logger="plsdo"):
            check_variance(arr, feature_names=["nearly_const", "varying"],
                            near_zero_threshold=0.85)
        assert "nearly_const" in caplog.text
        assert "near-zero variance" in caplog.text.lower()


class TestZscoreColumns:
    def test_zero_mean_unit_variance(self):
        arr = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        z = zscore_columns(arr)
        np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(z.std(axis=0, ddof=0), 1.0, atol=1e-10)

    def test_shape_preserved(self):
        arr = np.random.default_rng(0).standard_normal((10, 5))
        z = zscore_columns(arr)
        assert z.shape == arr.shape
