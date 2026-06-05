"""
Tests for src/aggregation.py — compute_hourly_stats, compute_daily_stats,
compute_daily_totals, and compute_summary.
"""
import pytest
import pandas as pd
from src.aggregation import (
    compute_hourly_stats,
    compute_daily_stats,
    compute_daily_totals,
    compute_summary,
)


def test_hourly_stats_shape(sample_df):
    result = compute_hourly_stats(sample_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_hourly_stats_has_expected_columns(sample_df):
    result = compute_hourly_stats(sample_df)
    for col in ("hour", "avg_kWh"):
        assert col in result.columns, f"Missing column: {col}"


def test_daily_stats_shape(sample_df):
    result = compute_daily_stats(sample_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_daily_totals_length(sample_df):
    result = compute_daily_totals(sample_df)
    unique_dates = sample_df["date"].nunique()
    assert len(result) == unique_dates


def test_daily_totals_no_negative_values(sample_df):
    result = compute_daily_totals(sample_df)
    assert (result["daily_kWh"] >= 0).all()


def test_summary_is_dict_or_series(sample_df):
    hourly = compute_hourly_stats(sample_df)
    daily = compute_daily_stats(sample_df)
    result = compute_summary(sample_df, hourly, daily)
    assert isinstance(result, (dict, pd.Series))


def test_summary_contains_mean_key(sample_df):
    hourly = compute_hourly_stats(sample_df)
    daily = compute_daily_stats(sample_df)
    result = compute_summary(sample_df, hourly, daily)
    keys = result.keys() if isinstance(result, dict) else result.index
    assert any("avg" in str(k).lower() for k in keys), \
        "Summary should include a mean consumption value"
