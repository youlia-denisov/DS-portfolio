"""
Tests for src/preprocessing.py — clean_consumption_data().

The function is expected to:
  - Accept a raw DataFrame and return a cleaned one
  - Drop rows where consumption_kwh is NaN or negative
  - Add 'hour', 'day_of_week', and 'date' columns if they are missing
  - Parse the timestamp column into datetime
"""
import pytest
import pandas as pd
import numpy as np
from src.preprocessing import clean_consumption_data


@pytest.fixture
def raw_df():
    """Minimal raw input as it might arrive from the CSV loader."""
    return pd.DataFrame({
        "timestamp": ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00",
                       "2024-01-01 03:00", "2024-01-01 04:00"],
        "consumption_kwh": [1.2, None, -0.5, 0.0, 2.1],
    })


def test_returns_dataframe(raw_df):
    result = clean_consumption_data(raw_df)
    assert isinstance(result, pd.DataFrame)


def test_drops_nan_consumption(raw_df):
    result = clean_consumption_data(raw_df)
    assert result["consumption_kwh"].isna().sum() == 0


def test_drops_negative_consumption(raw_df):
    result = clean_consumption_data(raw_df)
    assert (result["consumption_kwh"] < 0).sum() == 0


def test_derived_columns_present(raw_df):
    result = clean_consumption_data(raw_df)
    for col in ("hour", "day_of_week", "date"):
        assert col in result.columns, f"Missing column: {col}"


def test_hour_range(raw_df):
    result = clean_consumption_data(raw_df)
    assert result["hour"].between(0, 23).all()


def test_output_has_fewer_rows_than_input(raw_df):
    # raw_df has one NaN and one negative — both should be dropped
    result = clean_consumption_data(raw_df)
    assert len(result) < len(raw_df)
