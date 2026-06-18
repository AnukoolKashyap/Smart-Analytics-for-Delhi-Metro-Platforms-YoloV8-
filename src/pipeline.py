"""
pipeline.py
-----------
Ties detector -> zones -> heatmap -> analytics together and processes a
whole video end to end, writing:

    output/annotated_video.mp4   zone overlays + live counts + density colors
    output/heatmap.png           accumulated crowd-density heatmap
    output/analytics_raw.csv     per-frame zone occupancy log
    output/analytics_hourly.csv  bucketed (default hourly) aggregation
    output/summary.json          final redistribution suggestion + surge alerts

Run directly:
    python src/pipeline.py --video data/sample_platform_footage.avi --backend hog

(use --backend yolo for the real detector; downloads weights on first run)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta

import cv2
import imageio.v2 as imageio
import numpy as np

from analytics import AnalyticsLogger, bucket_aggregate, detect_surge, redistribution_suggestion
from detector import build_detector
from heatmap import HeatmapAccumulator
from zones import DENSITY_COLORS_BGR, ZoneManager

# Resolved relative to this file, not the caller's working directory, so
# `python src/pipeline.py ...` works the same whether you run it from the
# project root or from inside src/ -- a plain string default like
# "config/zones.json" would silently break in the second case.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ZONES_PATH = os.path.join(_PROJECT_ROOT, "config", "zones.json")


def parse_sim_start(s: str) -> int:
    """Parse 'HH:MM' or 'HH:MM:SS' into seconds-since-midnight."""
    parts = [int(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.append(0)
    h, m, sec = parts
    return h * 3600 + m * 60 + sec


def seconds_to_hhmm(total_seconds: float) -> str:
    total_seconds = int(total_seconds) % 86400
    return str(timedelta(seconds=total_seconds))[:-3] if total_seconds % 60 else str(timedelta(seconds=total_seconds))


def run_pipeline(
    video_path: str,
    zones_config_path: str = DEFAULT_ZONES_PATH,
    output_dir: str = "output",
    backend: str = "hog",
    model_path: str = "yolov8n.pt",
    device: str = "cpu",
    detect_every_n_frames: int = 2,
    bucket_seconds: int = 3600,
    sim_start_time: str = "08:00:00",
    sim_seconds_per_video_second: float = 60.0,
    max_frames: int = None,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)

    zone_mgr = ZoneManager(zones_config_path)
    zone_mgr.scale_to_frame(frame_w, frame_h)
    zone_ids = [z.id for z in zone_mgr.zones]
    zone_names = {z.id: z.name for z in zone_mgr.zones}
    capacities = {z.id: z.capacity for z in zone_mgr.zones}

    detector = build_detector(backend, model_path=model_path, device=device)
    heat = HeatmapAccumulator(frame_w, frame_h)
    logger = AnalyticsLogger()

    out_video_path = os.path.join(output_dir, "annotated_video.mp4")
    # cv2.VideoWriter's bundled FFmpeg generally can't encode H.264 (no licensed
    # encoder in most opencv-python builds) -- it silently falls back to mp4v,
    # which Chrome/Firefox/Streamlit's <video> tag won't play even though the
    # file itself is valid. imageio (with its bundled ffmpeg binary -- no
    # system ffmpeg install needed) writes real H.264/yuv420p, which is what
    # browsers actually need.
    writer = imageio.get_writer(
        out_video_path, fps=fps, codec="libx264", format="FFMPEG",
        pixelformat="yuv420p", macro_block_size=None,
    )

    sim_start_sec = parse_sim_start(sim_start_time)
    last_dets = []
    last_zone_counts = {zid: 0 for zid in zone_ids}

    frame_idx = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and frame_idx >= max_frames):
            break

        video_elapsed_sec = frame_idx / fps
        sim_timestamp_sec = sim_start_sec + video_elapsed_sec * sim_seconds_per_video_second

        if frame_idx % detect_every_n_frames == 0:
            dets = detector.detect(frame)
            last_dets = dets
            foot_points = [(d["id"], d["foot"][0], d["foot"][1]) for d in dets]
            zone_assignments = zone_mgr.assign(foot_points)
            last_zone_counts = {zid: len(ids) for zid, ids in zone_assignments.items()}
            heat.update([d["foot"] for d in dets])
            logger.record(sim_timestamp_sec, last_zone_counts)

        # --- draw overlay ---
        overlay = frame.copy()
        for z in zone_mgr.zones:
            count = last_zone_counts.get(z.id, 0)
            level = zone_mgr.density_level(count, z.capacity)
            color = DENSITY_COLORS_BGR[level]
            cv2.polylines(overlay, [z.polygon_px], isClosed=True, color=color, thickness=2)

        for d in last_dets:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 1)
            fx, fy = int(d["foot"][0]), int(d["foot"][1])
            cv2.circle(overlay, (fx, fy), 3, (255, 255, 255), -1)

        # legend panel (top-left): avoids the label overlap that happens when
        # floating text at each zone's centroid, since adjacent zones can
        # sit at the same vertical position
        clock_label = seconds_to_hhmm(sim_timestamp_sec)
        lines = [f"Sim time: {clock_label}"]
        line_colors = [(255, 255, 255)]
        for z in zone_mgr.zones:
            count = last_zone_counts.get(z.id, 0)
            level = zone_mgr.density_level(count, z.capacity)
            lines.append(f"{z.name}: {count}/{z.capacity} [{level}]")
            line_colors.append(DENSITY_COLORS_BGR[level])

        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        line_h = 22
        max_w = max(cv2.getTextSize(t, font, scale, thick)[0][0] for t in lines)
        panel_w, panel_h = max_w + 24, line_h * len(lines) + 14
        panel = overlay[8 : 8 + panel_h, 8 : 8 + panel_w]
        cv2.addWeighted(panel, 0.35, np.zeros_like(panel), 0.65, 0, dst=panel)
        for i, (text, color) in enumerate(zip(lines, line_colors)):
            y = 8 + 22 + i * line_h
            cv2.putText(overlay, text, (8 + 12, y), font, scale, color, thick, cv2.LINE_AA)

        writer.append_data(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        last_full_frame = frame
        frame_idx += 1

    cap.release()
    writer.close()
    elapsed = time.time() - t0
    print(f"Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx/max(elapsed,1e-6):.1f} fps)")

    # --- heatmap ---
    heatmap_path = os.path.join(output_dir, "heatmap.png")
    heat.save(last_full_frame, heatmap_path)

    # --- analytics export ---
    raw_df = logger.to_dataframe()
    raw_csv_path = os.path.join(output_dir, "analytics_raw.csv")
    raw_df.to_csv(raw_csv_path, index=False)

    bucketed = bucket_aggregate(raw_df, zone_ids, bucket_seconds=bucket_seconds)
    hourly_csv_path = os.path.join(output_dir, "analytics_hourly.csv")
    bucketed.to_csv(hourly_csv_path, index=False)

    final_counts = last_zone_counts
    suggestion = redistribution_suggestion(final_counts, capacities, zone_names)
    surges = detect_surge(bucketed, zone_ids, zone_names)

    summary = {
        "video": video_path,
        "backend": backend,
        "frames_processed": frame_idx,
        "fps_processing_speed": round(frame_idx / max(elapsed, 1e-6), 2),
        "final_zone_counts": final_counts,
        "final_density_levels": {
            zid: zone_mgr.density_level(final_counts.get(zid, 0), capacities[zid]) for zid in zone_ids
        },
        "redistribution_suggestion": suggestion,
        "surge_alerts": [s.__dict__ for s in surges],
        "outputs": {
            "annotated_video": out_video_path,
            "heatmap": heatmap_path,
            "analytics_raw_csv": raw_csv_path,
            "analytics_hourly_csv": hourly_csv_path,
        },
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delhi Metro crowd analytics pipeline")
    parser.add_argument("--video", required=True)
    parser.add_argument("--zones", default=DEFAULT_ZONES_PATH)
    parser.add_argument("--output", default="output")
    parser.add_argument("--backend", choices=["yolo", "hog"], default="hog")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--detect-every", type=int, default=2)
    parser.add_argument("--bucket-seconds", type=int, default=3600)
    parser.add_argument("--sim-start", default="08:00:00")
    parser.add_argument("--sim-seconds-per-video-second", type=float, default=60.0)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    summary = run_pipeline(
        video_path=args.video,
        zones_config_path=args.zones,
        output_dir=args.output,
        backend=args.backend,
        model_path=args.model,
        device=args.device,
        detect_every_n_frames=args.detect_every,
        bucket_seconds=args.bucket_seconds,
        sim_start_time=args.sim_start,
        sim_seconds_per_video_second=args.sim_seconds_per_video_second,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, indent=2))
