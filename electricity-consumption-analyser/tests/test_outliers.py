"""
Tests for src/outliers.py — detect_outliers_3sigma, detect_outliers_iqr,
and calculate_outlier_summary.

Both detectors should catch the extreme values injected in df_with_outliers.
"""
import pytest
import pandas as pd
from src.outliers import detect_outliers_3sigma, detect_outliers_iqr, calculate_outlier_summary


def test_3sigma_returns_dataframe(sample_df):
    result = detect_outliers_3sigma(sample_df)
    assert isinstance(result, pd.DataFrame)


def test_iqr_returns_dataframe(sample_df):
    result = detect_outliers_iqr(sample_df)
    assert isinstance(result, pd.DataFrame)


def test_3sigma_detects_injected_outliers(df_with_outliers):
    result = detect_outliers_3sigma(df_with_outliers)
    assert 999.0 in result["kWh"].values


def test_iqr_detects_injected_outliers(df_with_outliers):
    result = detect_outliers_iqr(df_with_outliers)
    assert 999.0 in result["kWh"].values


def test_no_outliers_on_clean_data(sample_df):
    result_3s = detect_outliers_3sigma(sample_df)
    result_iqr = detect_outliers_iqr(sample_df)
    assert len(result_3s) < len(sample_df) * 0.1, "Too many 3-sigma outliers in clean data"
    assert len(result_iqr) < len(sample_df) * 0.2, "Too many IQR outliers in clean data"


def test_outlier_summary_structure(df_with_outliers):
    out_3s = detect_outliers_3sigma(df_with_outliers)
    out_iqr = detect_outliers_iqr(df_with_outliers)
    summary = calculate_outlier_summary(df_with_outliers, out_3s, out_iqr)
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) > 0
