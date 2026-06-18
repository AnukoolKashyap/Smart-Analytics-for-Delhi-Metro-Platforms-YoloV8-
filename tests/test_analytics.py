"""
Tests for analytics.py. These mirror the exact worked examples in
ARCHITECTURE_AND_CODE_GUIDE.md, so the expected numbers below were
verified by actually running this code, not guessed.
"""

import pandas as pd
import pytest
from analytics import (
    AnalyticsLogger,
    bucket_aggregate,
    detect_surge,
    forecast_next_bucket,
    redistribution_suggestion,
)


def test_analytics_logger_records_rows_and_exports():
    logger = AnalyticsLogger()
    logger.record(0, {"zone_a": 2, "zone_b": 5})
    logger.record(30, {"zone_a": 3, "zone_b": 6})
    df = logger.to_dataframe()
    assert list(df["timestamp_sec"]) == [0, 30]
    assert list(df["zone_a"]) == [2, 3]


def test_bucket_aggregate_groups_by_time_window():
    raw = pd.DataFrame(
        [
            {"timestamp_sec": 0, "zone_a": 2, "zone_b": 5},
            {"timestamp_sec": 30, "zone_a": 3, "zone_b": 6},
            {"timestamp_sec": 65, "zone_a": 4, "zone_b": 9},
            {"timestamp_sec": 90, "zone_a": 2, "zone_b": 9},
        ]
    )
    result = bucket_aggregate(raw, ["zone_a", "zone_b"], bucket_seconds=60)
    assert len(result) == 2
    assert result.loc[0, "zone_a_mean"] == 2.5
    assert result.loc[0, "zone_b_mean"] == 5.5
    assert result.loc[1, "zone_a_mean"] == 3.0
    assert result.loc[1, "bucket_start_sec"] == 60


def test_bucket_aggregate_handles_empty_input():
    empty = pd.DataFrame(columns=["timestamp_sec"])
    result = bucket_aggregate(empty, ["zone_a"], bucket_seconds=60)
    assert result.empty


def test_redistribution_suggestion_flags_large_gap():
    msg = redistribution_suggestion(
        zone_counts={"zone_a": 9, "zone_b": 1},
        capacities={"zone_a": 10, "zone_b": 10},
        zone_names={"zone_a": "Zone A", "zone_b": "Zone B"},
    )
    assert msg is not None
    assert "Zone A" in msg and "Zone B" in msg


def test_redistribution_suggestion_silent_when_balanced():
    msg = redistribution_suggestion(
        zone_counts={"zone_a": 5, "zone_b": 5},
        capacities={"zone_a": 10, "zone_b": 10},
        zone_names={"zone_a": "Zone A", "zone_b": "Zone B"},
    )
    assert msg is None


def test_forecast_next_bucket_extrapolates_linear_trend():
    # slope is exactly +2 per step, so the next value should be 10
    assert forecast_next_bucket([2, 4, 6, 8]) == pytest.approx(10.0)


def test_forecast_next_bucket_needs_at_least_two_points():
    assert forecast_next_bucket([5]) is None
    assert forecast_next_bucket([]) is None


def test_forecast_never_goes_negative():
    # a sharply falling trend shouldn't forecast a negative crowd count
    assert forecast_next_bucket([10, 4, -2]) == 0.0


def test_detect_surge_flags_large_jump():
    bucketed = pd.DataFrame([{"zone_a_mean": 2, "zone_a_max": 3}, {"zone_a_mean": 7, "zone_a_max": 9}])
    alerts = detect_surge(bucketed, ["zone_a"], {"zone_a": "Zone A"}, threshold_pct=50)
    assert len(alerts) == 1
    assert alerts[0].pct_increase == pytest.approx(250.0)


def test_detect_surge_silent_when_stable():
    bucketed = pd.DataFrame([{"zone_a_mean": 5, "zone_a_max": 6}, {"zone_a_mean": 5.2, "zone_a_max": 6}])
    alerts = detect_surge(bucketed, ["zone_a"], {"zone_a": "Zone A"}, threshold_pct=50)
    assert alerts == []


def test_detect_surge_needs_at_least_two_buckets():
    bucketed = pd.DataFrame([{"zone_a_mean": 5, "zone_a_max": 6}])
    assert detect_surge(bucketed, ["zone_a"], {"zone_a": "Zone A"}) == []
