import numpy as np
import pandas as pd
import pytest
import yaml
from plsdo.io import (
    _normalise_sid,
    corrected_pvalue,
    load_csv,
    detect_subject_id,
    align_subjects,
    check_missing_values,
    check_variance,
    zscore_columns,
    parse_groups_config,
    GroupConfig,
    GroupSpec,
    load_metadata,
    build_design_matrix,
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

    def test_strips_leading_whitespace_in_headers(self, tmp_path):
        path = tmp_path / "spaced.csv"
        path.write_text("feature, category\nfoo,CBF\nbar,Hurst\n")
        df = load_csv(path, require_numeric=False)
        assert list(df.columns) == ["feature", "category"]

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
        sid = detect_subject_id([x_df, y_df, demographics_df], subject_id="subject_id")
        assert sid == ["subject_id"]

    def test_explicit_subject_id_missing_from_file(self, x_df, y_df):
        with pytest.raises(ValueError, match="not found"):
            detect_subject_id([x_df, y_df], subject_id="nonexistent")

    def test_auto_detect(self, x_df, y_df, demographics_df):
        sid = detect_subject_id([x_df, y_df, demographics_df])
        assert sid == ["subject_id"]

    def test_auto_detect_uses_positional_order_not_alphabetical(self):
        # "b" comes before "a" alphabetically but "a" is first in df1
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4], "val": [5, 6]})
        df2 = pd.DataFrame({"b": [3, 4], "a": [1, 2], "val2": [7, 8]})
        sid = detect_subject_id([df1, df2])
        assert sid == ["a"]

    def test_auto_detect_no_shared_column(self):
        df1 = pd.DataFrame({"a": [1], "val": [2]})
        df2 = pd.DataFrame({"b": [1], "val2": [3]})
        with pytest.raises(ValueError, match="No shared column"):
            detect_subject_id([df1, df2])


class TestAlignSubjects:
    def test_aligned_subjects_same_order(self, x_df, y_df, demographics_df):
        aligned = align_subjects([x_df, y_df, demographics_df], subject_id="subject_id")
        for df in aligned:
            assert list(df["subject_id"]) == [f"s{i:02d}" for i in range(1, 13)]

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

    def test_no_performance_warning_on_wide_frame(self):
        # Regression: reset_index on a wide frame used to trigger
        # pandas PerformanceWarning about fragmentation.
        import warnings

        rng = np.random.default_rng(0)
        n_subjects = 10
        n_features = 1000
        ids = [f"s{i:03d}" for i in range(n_subjects)]
        wide = pd.DataFrame(
            rng.standard_normal((n_subjects, n_features)),
            columns=[f"f{j}" for j in range(n_features)],
        )
        wide.insert(0, "id", ids)
        other = pd.DataFrame({"id": ids, "v": rng.standard_normal(n_subjects)})

        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.PerformanceWarning)
            aligned = align_subjects([wide, other], subject_id="id")
        assert list(aligned[0]["id"]) == sorted(ids)
        assert aligned[0].shape == (n_subjects, n_features + 1)


class TestNormaliseSid:
    def test_string_to_list(self):
        assert _normalise_sid("subject_id") == ["subject_id"]

    def test_list_passthrough(self):
        assert _normalise_sid(["subject_id", "run_id"]) == ["subject_id", "run_id"]

    def test_tuple_to_list(self):
        assert _normalise_sid(("subject_id", "run_id")) == ["subject_id", "run_id"]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            _normalise_sid([])

    def test_bad_type_raises(self):
        with pytest.raises(TypeError, match="string or list"):
            _normalise_sid(42)


class TestDetectSubjectIdMultiColumn:
    def test_explicit_multi_column(self, data_dir):
        brain = pd.read_csv(data_dir / "brain_multi.csv")
        behaviour = pd.read_csv(data_dir / "behaviour_multi.csv")
        sid = detect_subject_id([brain, behaviour], subject_id=["subject_id", "run_id"])
        assert sid == ["subject_id", "run_id"]

    def test_explicit_multi_column_missing(self, data_dir):
        brain = pd.read_csv(data_dir / "brain_multi.csv")
        behaviour = pd.read_csv(data_dir / "behaviour_multi.csv")
        with pytest.raises(ValueError, match="not found"):
            detect_subject_id(
                [brain, behaviour], subject_id=["subject_id", "nonexistent"]
            )

    def test_explicit_single_returns_list(self):
        df1 = pd.DataFrame({"id": [1], "v": [2]})
        df2 = pd.DataFrame({"id": [1], "w": [3]})
        sid = detect_subject_id([df1, df2], subject_id="id")
        assert sid == ["id"]
        assert isinstance(sid, list)

    def test_auto_detect_returns_list(self):
        df1 = pd.DataFrame({"id": [1], "v": [2]})
        df2 = pd.DataFrame({"id": [1], "w": [3]})
        sid = detect_subject_id([df1, df2])
        assert sid == ["id"]
        assert isinstance(sid, list)


class TestAlignSubjectsMultiColumn:
    def test_multi_column_alignment(self, data_dir):
        brain = pd.read_csv(data_dir / "brain_multi.csv")
        behaviour = pd.read_csv(data_dir / "behaviour_multi.csv")
        aligned = align_subjects(
            [brain, behaviour], subject_id=["subject_id", "run_id"]
        )
        assert len(aligned) == 2
        assert len(aligned[0]) == 12
        # Both dataframes should have the same compound key order
        keys_0 = list(
            aligned[0][["subject_id", "run_id"]].itertuples(index=False, name=None)
        )
        keys_1 = list(
            aligned[1][["subject_id", "run_id"]].itertuples(index=False, name=None)
        )
        assert keys_0 == keys_1

    def test_multi_column_drops_non_shared(self, caplog):
        df1 = pd.DataFrame(
            {"sid": ["a", "a", "b"], "run": [1, 2, 1], "v1": [10, 20, 30]}
        )
        df2 = pd.DataFrame(
            {"sid": ["a", "b", "b"], "run": [1, 1, 2], "v2": [40, 50, 60]}
        )
        with caplog.at_level("WARNING", logger="plsdo"):
            aligned = align_subjects([df1, df2], subject_id=["sid", "run"])
        # Only (a, 1) and (b, 1) are shared
        assert len(aligned[0]) == 2
        assert "not present in all files" in caplog.text

    def test_multi_column_empty_intersection_raises(self):
        df1 = pd.DataFrame(
            {"sid": ["a", "a"], "run": [1, 2], "v1": [10, 20]}
        )
        df2 = pd.DataFrame(
            {"sid": ["b", "b"], "run": [1, 2], "v2": [30, 40]}
        )
        with pytest.raises(ValueError, match="No subjects shared"):
            align_subjects([df1, df2], subject_id=["sid", "run"])

    def test_single_element_list_matches_string(self):
        df1 = pd.DataFrame({"id": ["a", "b", "c"], "v1": [1, 2, 3]})
        df2 = pd.DataFrame({"id": ["c", "a", "b"], "v2": [30, 10, 20]})
        aligned_str = align_subjects([df1.copy(), df2.copy()], subject_id="id")
        aligned_list = align_subjects(
            [df1.copy(), df2.copy()], subject_id=["id"]
        )
        for a, b in zip(aligned_str, aligned_list):
            pd.testing.assert_frame_equal(a, b)


class TestCheckMissingValues:
    def test_no_missing_passes(self):
        df = pd.DataFrame({"id": ["a", "b"], "v1": [1.0, 2.0], "v2": [3.0, 4.0]})
        check_missing_values(df, name="test")  # should not raise

    def test_missing_values_raises(self):
        df = pd.DataFrame(
            {"id": ["a", "b"], "v1": [1.0, float("nan")], "v2": [3.0, 4.0]}
        )
        with pytest.raises(ValueError, match="missing values"):
            check_missing_values(df, name="test")

    def test_reports_which_subjects_and_features(self):
        df = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "v1": [1.0, float("nan"), 3.0],
                "v2": [float("nan"), 2.0, 4.0],
            }
        )
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
            check_variance(
                arr, feature_names=["nearly_const", "varying"], near_zero_threshold=0.85
            )
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


class TestParseGroupsConfig:
    def test_loads_yaml(self, groups_yaml_path):
        config = parse_groups_config(groups_yaml_path)
        assert config.subject_id == "subject_id"
        assert len(config.groups) == 2
        assert config.groups[0].column == "group"
        assert config.groups[0].role == "x_axis"
        assert config.groups[0].reference == "A"
        assert config.groups[0].order == ["A", "B", "C"]

    def test_column_not_in_demographics_raises(self, tmp_path):
        cfg = tmp_path / "bad.yaml"
        cfg.write_text(
            yaml.dump(
                {
                    "subject_id": "subject_id",
                    "groups": [{"column": "nonexistent", "role": "x_axis"}],
                }
            )
        )
        demo = pd.DataFrame({"subject_id": ["a"], "group": ["A"]})
        with pytest.raises(ValueError, match="not found in demographics"):
            parse_groups_config(cfg, demographics_df=demo)

    def test_warns_about_unlisted_demographics_columns(self, tmp_path, caplog):
        cfg = tmp_path / "partial.yaml"
        cfg.write_text(
            yaml.dump(
                {
                    "subject_id": "subject_id",
                    "groups": [{"column": "group", "role": "x_axis"}],
                }
            )
        )
        demo = pd.DataFrame({"subject_id": ["a"], "group": ["A"], "extra_col": [1]})
        with caplog.at_level("WARNING", logger="plsdo"):
            parse_groups_config(cfg, demographics_df=demo)
        assert "extra_col" in caplog.text

    def test_from_group_col_string(self):
        config = GroupConfig.from_group_col("Drug")
        assert len(config.groups) == 1
        assert config.groups[0].column == "Drug"
        assert config.groups[0].role == "x_axis"

    def test_invalid_role_raises(self, tmp_path):
        cfg = tmp_path / "bad_role.yaml"
        cfg.write_text(
            yaml.dump(
                {
                    "subject_id": "id",
                    "groups": [{"column": "g", "role": "invalid_role"}],
                }
            )
        )
        with pytest.raises(ValueError, match="Invalid role"):
            parse_groups_config(cfg)

    def test_both_facet_roles_raises(self, tmp_path):
        """The latent variable already occupies one grid axis, so a config
        cannot request both facet_rows and facet_cols."""
        cfg = tmp_path / "both_facets.yaml"
        cfg.write_text(
            yaml.dump(
                {
                    "subject_id": "subject_id",
                    "groups": [
                        {"column": "group", "role": "x_axis"},
                        {"column": "sex", "role": "facet_rows"},
                        {"column": "site", "role": "facet_cols"},
                    ],
                }
            )
        )
        with pytest.raises(ValueError, match="both facet_rows and facet_cols"):
            parse_groups_config(cfg)

    def test_yaml_list_subject_id(self, tmp_path):
        cfg = tmp_path / "multi.yaml"
        cfg.write_text(
            yaml.dump(
                {
                    "subject_id": ["subject_id", "run_id"],
                    "groups": [{"column": "group", "role": "x_axis"}],
                }
            )
        )
        config = parse_groups_config(cfg)
        assert config.subject_id == ["subject_id", "run_id"]

    def test_unlisted_columns_excludes_multi_subject_id(self, tmp_path, caplog):
        cfg = tmp_path / "multi_sid.yaml"
        cfg.write_text(
            yaml.dump(
                {
                    "subject_id": ["subject_id", "run_id"],
                    "groups": [{"column": "group", "role": "x_axis"}],
                }
            )
        )
        demo = pd.DataFrame(
            {
                "subject_id": ["a"],
                "run_id": [1],
                "group": ["A"],
                "extra": [99],
            }
        )
        with caplog.at_level("WARNING", logger="plsdo"):
            parse_groups_config(cfg, demographics_df=demo)
        # subject_id and run_id should be excluded from the unlisted warning
        assert "subject_id" not in caplog.text
        assert "run_id" not in caplog.text
        assert "extra" in caplog.text


class TestLoadMetadata:
    def test_loads_valid_metadata(self, data_dir):
        meta = load_metadata(
            data_dir / "brain_meta.csv",
            data_feature_names=["x1", "x2", "x3", "x4", "x5"],
        )
        assert "feature" in meta.columns
        assert len(meta) == 5

    def test_feature_in_meta_not_in_data_raises(self, tmp_path):
        meta_path = tmp_path / "meta.csv"
        meta_path.write_text("feature,category\na,cat1\nb,cat2\nZZZ,cat3\n")
        with pytest.raises(ValueError, match="ZZZ"):
            load_metadata(meta_path, data_feature_names=["a", "b"])

    def test_feature_in_data_not_in_meta_warns(self, tmp_path, caplog):
        meta_path = tmp_path / "meta.csv"
        meta_path.write_text("feature,category\na,cat1\n")
        with caplog.at_level("WARNING", logger="plsdo"):
            load_metadata(meta_path, data_feature_names=["a", "b"])
        assert "b" in caplog.text
        assert "not in metadata" in caplog.text.lower()


class TestBuildDesignMatrix:
    def test_single_factor(self):
        demo = pd.DataFrame(
            {
                "subject_id": ["s1", "s2", "s3", "s4"],
                "group": ["A", "A", "B", "B"],
            }
        )
        config = GroupConfig.from_group_col("group")
        X, labels = build_design_matrix(demo, config)
        assert X.shape == (4, 2)  # 2 levels
        assert labels == ["group_A", "group_B"]
        # s1 and s2 should have [1, 0], s3 and s4 should have [0, 1]
        np.testing.assert_array_equal(X[0], [1, 0])
        np.testing.assert_array_equal(X[2], [0, 1])

    def test_multiple_factors_additive(self):
        demo = pd.DataFrame(
            {
                "subject_id": ["s1", "s2", "s3", "s4"],
                "geno": ["WT", "WT", "KO", "KO"],
                "drug": ["sal", "oxy", "sal", "oxy"],
            }
        )
        config = GroupConfig(
            groups=[
                GroupSpec(column="geno", role="x_axis"),
                GroupSpec(column="drug", role="hue"),
            ]
        )
        X, labels = build_design_matrix(demo, config)
        # 2 levels for geno + 2 levels for drug = 4 columns
        assert X.shape == (4, 4)
        assert labels == ["geno_KO", "geno_WT", "drug_oxy", "drug_sal"]

    def test_zero_variance_column_raises(self):
        demo = pd.DataFrame(
            {
                "subject_id": ["s1", "s2"],
                "group": ["A", "A"],  # only one level
            }
        )
        config = GroupConfig.from_group_col("group")
        with pytest.raises(ValueError, match="single level"):
            build_design_matrix(demo, config)

    def test_order_level_absent_from_data_raises(self):
        demo = pd.DataFrame(
            {
                "subject_id": ["s1", "s2", "s3"],
                "group": ["A", "A", "B"],  # C listed in order but not present
            }
        )
        config = GroupConfig(
            groups=[
                GroupSpec(column="group", role="x_axis", order=["A", "B", "C"]),
            ]
        )
        with pytest.raises(ValueError, match="zero variance"):
            build_design_matrix(demo, config)

    def test_reference_level_ordering(self):
        demo = pd.DataFrame(
            {
                "subject_id": ["s1", "s2", "s3"],
                "group": ["B", "A", "C"],
            }
        )
        config = GroupConfig(
            groups=[
                GroupSpec(
                    column="group", role="x_axis", reference="A", order=["A", "B", "C"]
                ),
            ]
        )
        X, labels = build_design_matrix(demo, config)
        assert labels == ["group_A", "group_B", "group_C"]

    def test_ignores_ignore_role(self):
        demo = pd.DataFrame(
            {
                "subject_id": ["s1", "s2"],
                "group": ["A", "B"],
                "cage": [1, 2],
            }
        )
        config = GroupConfig(
            groups=[
                GroupSpec(column="group", role="x_axis"),
                GroupSpec(column="cage", role="ignore"),
            ]
        )
        X, labels = build_design_matrix(demo, config)
        assert X.shape == (2, 2)  # only group, not cage

    def test_facet_roles_are_modelled(self):
        """Any role other than 'ignore' puts the factor in the model: a
        facet_rows/facet_cols column is dummy-coded into the design matrix
        alongside x_axis and hue. ('ignore' is the only exclusion.)"""
        demo = pd.DataFrame(
            {
                "subject_id": ["s1", "s2", "s3", "s4"],
                "geno": ["A", "A", "B", "B"],
                "sex": ["F", "M", "F", "M"],
            }
        )
        config = GroupConfig(
            groups=[
                GroupSpec(column="geno", role="x_axis"),
                GroupSpec(column="sex", role="facet_rows"),
            ]
        )
        X, labels = build_design_matrix(demo, config)
        # Both factors contribute dummy columns to the additive design.
        assert labels == ["geno_A", "geno_B", "sex_F", "sex_M"]
        assert X.shape == (4, 4)


class TestGroupConfigRoleQueries:
    def test_active_groups_excludes_ignore(self):
        config = GroupConfig(
            groups=[
                GroupSpec(column="group", role="x_axis"),
                GroupSpec(column="cage", role="ignore"),
                GroupSpec(column="sex", role="hue"),
            ]
        )
        cols = [g.column for g in config.active_groups()]
        assert cols == ["group", "sex"]

    def test_x_axis_group_prefers_explicit_role(self):
        config = GroupConfig(
            groups=[
                GroupSpec(column="sex", role="hue"),
                GroupSpec(column="group", role="x_axis"),
            ]
        )
        assert config.x_axis_group().column == "group"

    def test_x_axis_group_falls_back_to_first_active(self):
        """No explicit x_axis role -> first non-ignore group."""
        config = GroupConfig(
            groups=[
                GroupSpec(column="cage", role="ignore"),
                GroupSpec(column="sex", role="hue"),
                GroupSpec(column="batch", role="hue"),
            ]
        )
        assert config.x_axis_group().column == "sex"

    def test_x_axis_group_none_when_all_ignored(self):
        config = GroupConfig(groups=[GroupSpec(column="cage", role="ignore")])
        assert config.x_axis_group() is None

    def test_hue_column_returns_hue_role(self):
        config = GroupConfig(
            groups=[
                GroupSpec(column="group", role="x_axis"),
                GroupSpec(column="sex", role="hue"),
            ]
        )
        assert config.hue_column() == "sex"

    def test_hue_column_none_when_absent(self):
        config = GroupConfig(groups=[GroupSpec(column="group", role="x_axis")])
        assert config.hue_column() is None

    def test_facet_rows_and_cols_columns(self):
        config = GroupConfig(
            groups=[
                GroupSpec(column="group", role="x_axis"),
                GroupSpec(column="sex", role="facet_rows"),
                GroupSpec(column="site", role="facet_cols"),
            ]
        )
        assert config.facet_rows_column() == "sex"
        assert config.facet_cols_column() == "site"

    def test_facet_col_wrap_read_from_any_active_group(self):
        # facet_col_wrap applies to the default LV-on-columns layout, so it is
        # set on the model group rather than a facet group.
        config = GroupConfig(
            groups=[GroupSpec(column="group", role="x_axis", facet_col_wrap=3)]
        )
        assert config.facet_col_wrap() == 3

    def test_facet_columns_none_when_absent(self):
        config = GroupConfig(groups=[GroupSpec(column="group", role="x_axis")])
        assert config.facet_rows_column() is None
        assert config.facet_cols_column() is None
        assert config.facet_col_wrap() is None


class TestCorrectedPvalue:
    def test_observed_exceeds_all_null(self):
        """Nothing in the null beats the observed -> minimum p = 1/(n+1)."""
        null = np.arange(20.0)
        assert corrected_pvalue(100.0, null) == pytest.approx(1 / 21)

    def test_observed_below_all_null(self):
        """Everything in the null beats the observed -> p = 1.0."""
        null = np.arange(1.0, 21.0)
        assert corrected_pvalue(0.0, null) == pytest.approx(1.0)

    def test_correction_never_zero(self):
        null = np.zeros(10)
        assert corrected_pvalue(1e9, null) > 0.0

    def test_counts_ties_as_exceeding(self):
        """>= is inclusive: a null value equal to observed counts."""
        null = np.array([1.0, 2.0, 2.0, 3.0])
        # three values >= 2.0 (both 2s and the 3) -> (3 + 1) / (4 + 1)
        assert corrected_pvalue(2.0, null) == pytest.approx(4 / 5)

    def test_vectorised_matches_scalar(self):
        """Row-wise application (one null row per LV) matches scalar calls."""
        rng = np.random.default_rng(0)
        observed = np.array([2.0, 0.5, 1.0])
        null = rng.standard_normal((3, 50))
        vec = corrected_pvalue(observed, null, axis=1)
        for i in range(3):
            assert vec[i] == pytest.approx(corrected_pvalue(observed[i], null[i]))
