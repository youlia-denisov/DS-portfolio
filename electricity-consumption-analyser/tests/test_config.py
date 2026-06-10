"""
Tests for config.py — verifies that all required paths and constants are
defined with sensible types and values.
"""
import pytest
from pathlib import Path
import config


def test_all_dirs_are_path_objects():
    dirs = [
        config.DATA_DIR, config.RAW_DIR, config.PROCESSED_DIR,
        config.OUTPUT_DIR, config.HTML_DIR, config.FIGURE_DIR,
        config.TABLE_DIR, config.REPORT_DIR,
    ]
    for path in dirs:
        assert isinstance(path, Path), f"{path} should be a pathlib.Path"


def test_consumption_file_is_csv():
    assert config.CONSUMPTION_FILE.suffix == ".csv"


def test_tariff_is_positive_float():
    assert isinstance(config.TARIFF, float)
    assert config.TARIFF > 0, "Tariff must be a positive price per kWh"


def test_weekday_order_has_seven_days():
    assert len(config.WEEKDAY_ORDER) == 7


def test_clustering_constants():
    assert isinstance(config.N_CLUSTERS, int)
    assert config.N_CLUSTERS >= 2, "Need at least 2 clusters"
    assert isinstance(config.RANDOM_STATE, int)


def test_has_smart_meter_is_none_or_bool():
    assert config.HAS_SMART_METER in (None, True, False)
