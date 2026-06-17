"""
detector.py
-----------
Pluggable person-detection + tracking backends. Both backends return the
same shape of output so the rest of the pipeline doesn't care which one is
running:

    [{"id": "3", "bbox": (x1, y1, x2, y2), "foot": (cx, y2)}, ...]

"foot" is the bottom-center of the bounding box (where the person is
standing), used for zone assignment and the heatmap instead of the box
center -- it's a much better approximation of ground position for a
crowd-density read, since head/torso pixel position drifts a lot with
camera angle and person height.

Backends
--------
YOLODetector   Primary backend. Uses Ultralytics YOLOv8 with built-in
                ByteTrack ("model.track(...)") for both detection and
                identity tracking. Requires the model weights, which are
                downloaded automatically on first run (needs an internet
                connection once; cached afterwards).

HOGDetector     Fully offline fallback using OpenCV's built-in HOG person
                detector + a lightweight centroid tracker implemented
                below. Lower accuracy and slower per-frame, but useful for
                quick smoke-testing the pipeline on a machine/sandbox with
                no internet access, or as a dependency-free baseline.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np


class CentroidTracker:
    """Minimal nearest-centroid tracker (no Kalman filter, no re-ID model).

    Good enough to give the HOG fallback stable IDs across frames for
    short-range tracking; not a substitute for ByteTrack/DeepSORT on the
    YOLO path."""

    def __init__(self, max_distance: int = 60, max_missed: int = 10):
        self.next_id = 0
        self.objects: Dict[int, Tuple[float, float]] = {}
        self.missed: Dict[int, int] = {}
        self.max_distance = max_distance
        self.max_missed = max_missed

    def update(self, points: List[Tuple[float, float]]) -> Dict[int, Tuple[float, float]]:
        if not self.objects:
            for p in points:
                self.objects[self.next_id] = p
                self.missed[self.next_id] = 0
                self.next_id += 1
            return dict(self.objects)

        existing_ids = list(self.objects.keys())
        existing_pts = np.array([self.objects[i] for i in existing_ids], dtype=np.float32)
        used_existing = set()
        used_new = set()

        if points:
            new_pts = np.array(points, dtype=np.float32)
            dists = np.linalg.norm(existing_pts[:, None, :] - new_pts[None, :, :], axis=2)
            pairs = [
                (dists[i, j], i, j)
                for i in range(dists.shape[0])
                for j in range(dists.shape[1])
            ]
            pairs.sort(key=lambda t: t[0])
            for dist, i, j in pairs:
                if i in used_existing or j in used_new:
                    continue
                if dist > self.max_distance:
                    continue
                oid = existing_ids[i]
                self.objects[oid] = tuple(points[j])
                self.missed[oid] = 0
                used_existing.add(i)
                used_new.add(j)

        for i, oid in enumerate(existing_ids):
            if i not in used_existing:
                self.missed[oid] += 1
                if self.missed[oid] > self.max_missed:
                    del self.objects[oid]
                    del self.missed[oid]

        for j, p in enumerate(points):
            if j not in used_new:
                self.objects[self.next_id] = p
                self.missed[self.next_id] = 0
                self.next_id += 1

        return dict(self.objects)


class HOGDetector:
    """Offline fallback person detector. No model download required."""

    def __init__(self, hit_threshold: float = 0.0):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.hit_threshold = hit_threshold
        self.tracker = CentroidTracker()

    def detect(self, frame: np.ndarray) -> List[dict]:
        # HOG is tuned for ~64x128 windows; downscale large frames for speed.
        h, w = frame.shape[:2]
        scale = 1.0
        max_w = 640
        if w > max_w:
            scale = max_w / w
            frame_small = cv2.resize(frame, (max_w, int(h * scale)))
        else:
            frame_small = frame

        rects, weights = self.hog.detectMultiScale(
            frame_small, winStride=(8, 8), padding=(8, 8), scale=1.05
        )

        boxes = []
        for (x, y, bw, bh), wt in zip(rects, weights):
            if wt < self.hit_threshold:
                continue
            x1, y1, x2, y2 = x / scale, y / scale, (x + bw) / scale, (y + bh) / scale
            boxes.append((x1, y1, x2, y2))

        foot_points = [((x1 + x2) / 2, y2) for (x1, y1, x2, y2) in boxes]
        tracked = self.tracker.update(foot_points)

        # match tracked ids back to nearest box (tracker only stores points)
        id_by_point = {v: k for k, v in tracked.items()}
        results = []
        for box, fp in zip(boxes, foot_points):
            # tracker may have merged/aged points; find closest tracked id
            best_id, best_d = None, float("inf")
            for tid, tp in tracked.items():
                d = (tp[0] - fp[0]) ** 2 + (tp[1] - fp[1]) ** 2
                if d < best_d:
                    best_d, best_id = d, tid
            results.append({"id": str(best_id), "bbox": box, "foot": fp})
        return results


class YOLODetector:
    """Primary backend: YOLOv8 detection + ByteTrack identity tracking."""

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35, device: str = "cpu"):
        from ultralytics import YOLO  # imported lazily so HOG-only use needs no torch/ultralytics issues

        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def detect(self, frame: np.ndarray) -> List[dict]:
        result = self.model.track(
            frame,
            persist=True,
            classes=[0],  # COCO class 0 = person
            conf=self.conf,
            device=self.device,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        results = []
        if result.boxes is None or result.boxes.id is None:
            return results

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy()
        for (x1, y1, x2, y2), tid in zip(boxes, ids):
            foot = ((x1 + x2) / 2, y2)
            results.append({"id": str(int(tid)), "bbox": (x1, y1, x2, y2), "foot": foot})
        return results


def build_detector(backend: str, model_path: str = "yolov8n.pt", device: str = "cpu"):
    if backend == "yolo":
        return YOLODetector(model_path=model_path, device=device)
    if backend == "hog":
        return HOGDetector()
    raise ValueError(f"Unknown detector backend: {backend}")
