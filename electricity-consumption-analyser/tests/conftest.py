"""
Shared fixtures for all test modules.

The synthetic DataFrame mimics the structure produced by clean_consumption_data():
  - 'datetime' column
  - 'kWh' column (float)
  - 'hour', 'weekday', 'date' columns
"""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_df():
    """48 hours of synthetic hourly consumption data covering two full days."""
    rng = np.random.default_rng(42)
    n = 48

    timestamps = pd.date_range("2024-01-01", periods=n, freq="h")
    consumption = rng.uniform(0.1, 3.0, size=n)

    df = pd.DataFrame({
        "datetime": timestamps,
        "kWh": consumption,
    })
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.day_name()
    df["date"] = df["datetime"].dt.date
    return df


@pytest.fixture
def df_with_outliers(sample_df):
    """Same as sample_df but with two clearly extreme values injected."""
    df = sample_df.copy()
    df.loc[5,  "kWh"] = 999.0   # extreme high
    df.loc[10, "kWh"] = -50.0   # extreme low (invalid)
    return df
