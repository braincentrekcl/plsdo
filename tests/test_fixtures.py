def test_data_files_exist(data_dir):
    assert (data_dir / "brain.csv").exists()
    assert (data_dir / "behaviour.csv").exists()
    assert (data_dir / "demographics.csv").exists()


def test_x_shape(x_df):
    assert x_df.shape == (12, 6)  # 12 subjects, subject_id + 5 features


def test_y_shape(y_df):
    assert y_df.shape == (12, 5)  # 12 subjects, subject_id + 4 features


def test_demographics_shape(demographics_df):
    assert demographics_df.shape == (12, 3)  # 12 subjects, subject_id + group + sex
