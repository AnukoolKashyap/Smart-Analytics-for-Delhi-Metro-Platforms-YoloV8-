"""
heatmap.py
----------
Accumulates person foot-positions over the whole video into a 2D density
map, then renders it as a JET-colormap overlay on a reference frame -- the
classic "where does the crowd actually pool on this platform" visualization.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


class HeatmapAccumulator:
    def __init__(self, frame_w: int, frame_h: int, blur_kernel: int = 31):
        self.w = frame_w
        self.h = frame_h
        self.accum = np.zeros((frame_h, frame_w), dtype=np.float32)
        # kernel must be odd
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1

    def update(self, foot_points: List[Tuple[float, float]]) -> None:
        for x, y in foot_points:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < self.w and 0 <= yi < self.h:
                self.accum[yi, xi] += 1.0

    def render(self, base_frame: np.ndarray, alpha: float = 0.55) -> np.ndarray:
        smoothed = cv2.GaussianBlur(self.accum, (self.blur_kernel, self.blur_kernel), 0)
        if smoothed.max() > 0:
            norm = (smoothed / smoothed.max() * 255).astype(np.uint8)
        else:
            norm = smoothed.astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        # only tint areas with non-trivial accumulated density so empty
        # platform space stays close to the original frame, not solid blue
        mask = (norm > 8).astype(np.uint8)[..., None]
        blended = base_frame.copy()
        overlay = cv2.addWeighted(base_frame, 1 - alpha, colored, alpha, 0)
        blended = np.where(mask > 0, overlay, blended)
        return blended

    def save(self, base_frame: np.ndarray, path: str, alpha: float = 0.55) -> None:
        img = self.render(base_frame, alpha=alpha)
        cv2.imwrite(path, img)
