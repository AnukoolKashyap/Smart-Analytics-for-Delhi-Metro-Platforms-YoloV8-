"""
Tests for detector.py's CentroidTracker -- the nearest-centroid tracker
backing the offline HOG path. No video or model weights needed; this is
pure point-matching logic, fed synthetic coordinates frame by frame.
"""

from detector import CentroidTracker


def test_tracker_assigns_sequential_ids_on_first_frame():
    tracker = CentroidTracker()
    result = tracker.update([(10, 10), (50, 50)])
    assert result == {0: (10, 10), 1: (50, 50)}


def test_tracker_keeps_same_id_for_a_moving_point():
    tracker = CentroidTracker(max_distance=60)
    tracker.update([(100, 200)])
    tracker.update([(110, 200)])
    result = tracker.update([(120, 200)])
    assert result == {0: (120, 200)}


def test_tracker_assigns_new_id_to_a_newcomer():
    tracker = CentroidTracker(max_distance=60)
    tracker.update([(100, 200)])
    result = tracker.update([(110, 200), (400, 400)])
    assert result[0] == (110, 200)
    assert result[1] == (400, 400)


def test_tracker_does_not_merge_points_beyond_max_distance():
    """A point that jumps further than max_distance should NOT be treated
    as the same object -- it should be dropped as a missed frame for the
    old ID, and the new point should get its own fresh ID."""
    tracker = CentroidTracker(max_distance=20, max_missed=5)
    tracker.update([(0, 0)])
    result = tracker.update([(500, 500)])
    assert 1 in result  # the far-away point became a new track
    assert result[1] == (500, 500)


def test_tracker_forgets_object_after_too_many_missed_frames():
    tracker = CentroidTracker(max_distance=60, max_missed=2)
    tracker.update([(10, 10)])
    tracker.update([])  # missed 1
    tracker.update([])  # missed 2
    result = tracker.update([])  # missed 3 -> should be forgotten now
    assert result == {}
