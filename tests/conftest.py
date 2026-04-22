"""Shared test fixtures."""

import pandas as pd
import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def data_dir():
    return DATA_DIR


@pytest.fixture
def x_df():
    return pd.read_csv(DATA_DIR / "brain.csv")


@pytest.fixture
def y_df():
    return pd.read_csv(DATA_DIR / "behaviour.csv")


@pytest.fixture
def demographics_df():
    return pd.read_csv(DATA_DIR / "demographics.csv")


@pytest.fixture
def x_meta_df():
    return pd.read_csv(DATA_DIR / "brain_meta.csv")


@pytest.fixture
def y_meta_df():
    return pd.read_csv(DATA_DIR / "behaviour_meta.csv")


@pytest.fixture
def groups_yaml_path():
    return DATA_DIR / "groups.yaml"


@pytest.fixture
def x_array(x_df):
    """Numeric X matrix as numpy array (no subject ID column)."""
    return x_df.drop(columns=["subject_id"]).to_numpy(dtype=float)


@pytest.fixture
def y_array(y_df):
    """Numeric Y matrix as numpy array (no subject ID column)."""
    return y_df.drop(columns=["subject_id"]).to_numpy(dtype=float)


@pytest.fixture
def tmp_output(tmp_path):
    """Temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out
