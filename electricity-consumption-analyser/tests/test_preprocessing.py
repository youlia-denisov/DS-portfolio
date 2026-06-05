"""
Tests for src/preprocessing.py — clean_consumption_data().

The function is expected to:
  - Accept a raw DataFrame with 'date', 'time', and 'kWh' columns
  - Return a cleaned DataFrame with NaN and negative kWh values handled
  - Add 'hour', 'weekday', 'month', 'is_weekend', and 'datetime' columns
"""
import pytest
import pandas as pd
from src.preprocessing import clean_consumption_data


@pytest.fixture
def raw_df():
    """Minimal raw input mimicking a real IEC CSV file."""
    return pd.DataFrame({
        "date": ["01/01/2024", "01/01/2024", "01/01/2024", "01/01/2024", "01/01/2024"],
        "time": ["00:00", "01:00", "02:00", "03:00", "04:00"],
        "kWh": ["1.2", "", "-0.5", "0.0", "2.1"],
    })


def test_returns_dataframe(raw_df):
    result = clean_consumption_data(raw_df)
    assert isinstance(result, pd.DataFrame)


def test_no_nan_consumption(raw_df):
    result = clean_consumption_data(raw_df)
    assert result["kWh"].isna().sum() == 0


def test_derived_columns_present(raw_df):
    result = clean_consumption_data(raw_df)
    for col in ("datetime", "hour", "weekday", "month", "is_weekend"):
        assert col in result.columns, f"Missing column: {col}"


def test_hour_range(raw_df):
    result = clean_consumption_data(raw_df)
    assert result["hour"].between(0, 23).all()


def test_datetime_is_parsed(raw_df):
    result = clean_consumption_data(raw_df)
    assert pd.api.types.is_datetime64_any_dtype(result["datetime"])
