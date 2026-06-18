"""
Tests for zones.py: normalized-to-pixel scaling, point-in-polygon
assignment, and density classification. No video/model dependency --
these are pure geometry and arithmetic, so they run in milliseconds.
"""

import json
import os
import tempfile

import pytest
from zones import ZoneManager


@pytest.fixture
def zones_config_path():
    """A small, predictable 2-zone config (left half / right half) so
    expected results are obvious, rather than reusing the shipped
    config/zones.json and having to recompute thirds by hand."""
    config = {
        "zones": [
            {"id": "left", "name": "Left Half", "polygon": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]], "capacity": 10},
            {"id": "right", "name": "Right Half", "polygon": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]], "capacity": 10},
        ],
        "density_thresholds": {"low": 0.4, "medium": 0.7, "high": 0.9},
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(config, f)
    yield path
    os.remove(path)


def test_scale_to_frame_converts_normalized_to_pixels(zones_config_path):
    zm = ZoneManager(zones_config_path)
    zm.scale_to_frame(200, 100)  # width=200, height=100
    left = zm.zone_by_id("left")
    # 0.5 of width=200 should land exactly at pixel x=100
    xs = [pt[0] for pt in left.polygon_px]
    assert max(xs) == 100
    assert min(xs) == 0


def test_assign_splits_points_into_correct_zones(zones_config_path):
    zm = ZoneManager(zones_config_path)
    zm.scale_to_frame(200, 100)
    points = [("p1", 10, 50), ("p2", 190, 50), ("p3", 99, 50)]
    result = zm.assign(points)
    assert result["left"] == ["p1", "p3"]
    assert result["right"] == ["p2"]


def test_assign_drops_points_outside_every_zone(zones_config_path):
    # A point with a negative coordinate falls outside both halves.
    zm = ZoneManager(zones_config_path)
    zm.scale_to_frame(200, 100)
    result = zm.assign([("ghost", -50, 50)])
    assert result["left"] == []
    assert result["right"] == []


@pytest.mark.parametrize(
    "count,capacity,expected",
    [
        (0, 10, "LOW"),
        (3, 10, "LOW"),     # ratio 0.3 < 0.4
        (5, 10, "MEDIUM"),  # ratio 0.5, between 0.4 and 0.7
        (8, 10, "HIGH"),    # ratio 0.8, between 0.7 and 0.9
        (10, 10, "CRITICAL"),  # ratio 1.0
        (0, 0, "LOW"),      # zero-capacity edge case shouldn't divide by zero
    ],
)
def test_density_level_thresholds(zones_config_path, count, capacity, expected):
    zm = ZoneManager(zones_config_path)
    assert zm.density_level(count, capacity) == expected


def test_scale_to_frame_is_cached_and_idempotent(zones_config_path):
    """Calling scale_to_frame twice with the same size shouldn't recompute
    (this is what makes calling it once per video frame cheap)."""
    zm = ZoneManager(zones_config_path)
    zm.scale_to_frame(200, 100)
    first = zm.zones[0].polygon_px.copy()
    zm.scale_to_frame(200, 100)
    second = zm.zones[0].polygon_px
    assert (first == second).all()
