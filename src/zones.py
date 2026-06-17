"""
zones.py
--------
Loads platform-zone definitions from a JSON config and provides utilities to:
  - scale normalized zone polygons to a given frame's pixel size
  - test which zone a detected person's foot-point falls into
  - classify a zone's crowd density (LOW / MEDIUM / HIGH / CRITICAL) against
    its configured capacity

Zones are stored in NORMALIZED coordinates (0-1 range) so the same config
file works for any camera resolution. Use tools/zone_selector.py to draw
real platform zones on your own footage instead of the default thirds-split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

DensityLevel = str  # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"


@dataclass
class Zone:
    id: str
    name: str
    polygon_norm: List[Tuple[float, float]]
    capacity: int
    polygon_px: np.ndarray = field(default=None, repr=False)  # filled in once scaled

    def scale_to(self, frame_w: int, frame_h: int) -> None:
        pts = [(int(x * frame_w), int(y * frame_h)) for x, y in self.polygon_norm]
        self.polygon_px = np.array(pts, dtype=np.int32)

    def contains_point(self, x: float, y: float) -> bool:
        if self.polygon_px is None:
            raise RuntimeError(f"Zone '{self.id}' was never scaled to a frame size.")
        return cv2.pointPolygonTest(self.polygon_px, (float(x), float(y)), False) >= 0

    def centroid_px(self) -> Tuple[int, int]:
        M = self.polygon_px.mean(axis=0)
        return int(M[0]), int(M[1])


class ZoneManager:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            cfg = json.load(f)

        self.zones: List[Zone] = [
            Zone(
                id=z["id"],
                name=z["name"],
                polygon_norm=[tuple(p) for p in z["polygon"]],
                capacity=int(z["capacity"]),
            )
            for z in cfg["zones"]
        ]
        th = cfg.get("density_thresholds", {"low": 0.4, "medium": 0.7, "high": 0.9})
        self.low_th = th["low"]
        self.medium_th = th["medium"]
        self.high_th = th["high"]
        self._scaled_for = None  # (w, h) cache so we don't rescale every frame

    def scale_to_frame(self, frame_w: int, frame_h: int) -> None:
        if self._scaled_for == (frame_w, frame_h):
            return
        for z in self.zones:
            z.scale_to(frame_w, frame_h)
        self._scaled_for = (frame_w, frame_h)

    def assign(self, points: List[Tuple[str, float, float]]) -> Dict[str, List[str]]:
        """
        points: list of (track_id, x, y) foot-point coordinates in pixels.
        returns: dict zone_id -> list of track_ids currently inside that zone.
        Points that fall outside every zone are dropped (e.g. someone on the
        tracks/concourse edge, outside the defined platform area).
        """
        result: Dict[str, List[str]] = {z.id: [] for z in self.zones}
        for track_id, x, y in points:
            for z in self.zones:
                if z.contains_point(x, y):
                    result[z.id].append(track_id)
                    break  # zones are assumed non-overlapping
        return result

    def density_level(self, count: int, capacity: int) -> DensityLevel:
        ratio = count / capacity if capacity > 0 else 0.0
        if ratio < self.low_th:
            return "LOW"
        if ratio < self.medium_th:
            return "MEDIUM"
        if ratio < self.high_th:
            return "HIGH"
        return "CRITICAL"

    def zone_by_id(self, zone_id: str) -> Zone:
        return next(z for z in self.zones if z.id == zone_id)


DENSITY_COLORS_BGR = {
    "LOW": (80, 200, 80),
    "MEDIUM": (0, 200, 230),
    "HIGH": (0, 120, 255),
    "CRITICAL": (0, 0, 255),
}
