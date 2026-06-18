# Architecture

This system turns a platform-facing video feed into zone-level crowd analytics: person detection and tracking feed a zone-assignment layer, which in turn feeds both a spatial heatmap and a time-series analytics layer. Everything below is produced by `pipeline.py` orchestrating the four modules in `src/`.

## Pipeline overview

```
                         ┌──────────────────────┐
   video file  ───────►  │   detector.py        │   "find the people"
                         │   (YOLODetector or   │   → list of {id, bbox, foot}
                         │    HOGDetector)      │
                         └───────────┬──────────┘
                                     │  foot-points (where each person is standing)
                                     ▼
                         ┌──────────────────────┐
                         │   zones.py           │   "which named region is
                         │   (ZoneManager)      │    each person standing in?
                         └───────────┬──────────┘
                                     │  per-zone counts
                       ┌─────────────┴─────────────┐
                       ▼                             ▼
          ┌────────────────────┐         ┌───────────────────────┐
          │  heatmap.py        │         │  analytics.py         │
          │  "where do people  │         │  "log it, bucket it,  │
          │   pool over time?" │         │   spot trends/alerts  │
          └────────────────────┘         └───────────────────────┘
                       │                             │
                       └─────────────┬───────────────┘
                                     ▼
                         ┌──────────────────────┐
                         │   pipeline.py        │   orchestrates all of the
                         │   (run_pipeline)     │   above, draws the overlay,
                         │                      │   writes every output file
                         └───────────┬──────────┘
                                     ▼
                annotated_video.mp4, heatmap.png,
                analytics_raw.csv, analytics_hourly.csv, summary.json
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │   dashboard/app.py   │   Streamlit UI that calls
                         │                      │   run_pipeline() and displays
                         │                      │   everything it produces
                         └──────────────────────┘
```

## How one frame moves through the system

1. `pipeline.py` reads one frame from the video.
2. The active detector (`YOLODetector` or `HOGDetector`) returns a list of `{id, bbox, foot}` for every person found — `foot` is the bottom-center of the box, used as the ground-position estimate instead of the box center.
3. `ZoneManager.assign()` checks each foot-point against every zone polygon (`cv2.pointPolygonTest`) and returns a per-zone list of which IDs are inside.
4. Each zone's count is compared against its configured capacity to produce a `LOW / MEDIUM / HIGH / CRITICAL` label.
5. The same foot-points are added to a running `HeatmapAccumulator`.
6. The per-zone counts for this frame are appended to the analytics log, timestamped on a (possibly time-scaled) simulated clock.
7. `pipeline.py` draws the zone outlines, detection boxes, and a status legend onto the frame and writes it to the output video.

## Module breakdown

**`src/detector.py`** — Two interchangeable detector backends behind one interface. `YOLODetector` wraps Ultralytics YOLOv8 with built-in ByteTrack (`model.track(...)`) for both detection and identity tracking; it needs model weights downloaded once. `HOGDetector` is a fully offline fallback using OpenCV's built-in HOG person detector, paired with a small hand-written `CentroidTracker` (nearest-centroid matching with a max-distance cutoff and a missed-frame timeout) since HOG has no tracking of its own. `build_detector(backend)` is the factory that hands back whichever one's requested, so nothing downstream needs to know which is running.

**`src/zones.py`** — `Zone` holds one polygon plus a capacity; polygons are stored as normalized (0–1) coordinates so the same config works at any camera resolution, and get scaled to actual pixels once per frame size via `scale_to_frame()` (cached, so it's not redone every frame). `ZoneManager` owns the full zone list and exposes `assign()` (point-in-polygon test per zone) and `density_level()` (count/capacity ratio against configurable thresholds).

**`src/heatmap.py`** — `HeatmapAccumulator` keeps a running 2D grid that increments at every foot-point's location. At render time (once, at the end of the run — not per frame) it Gaussian-blurs the grid to turn sharp points into soft blobs, applies a JET colormap, and blends only over non-trivial-density regions so empty platform space stays close to the original frame instead of being tinted solid blue.

**`src/analytics.py`** — Pure data-processing, no CV involved. `AnalyticsLogger` records per-frame zone counts and exports to CSV/JSON. `bucket_aggregate()` groups the raw log into fixed time windows (mean/max per zone) — set to 3600s for true hourly analytics on real footage. `redistribution_suggestion()` compares zone occupancy ratios and proposes redirecting passengers when the gap exceeds a threshold. `forecast_next_bucket()` is a naive linear-trend extrapolation over the last few buckets, deliberately simple rather than a trained time-series model. `detect_surge()` flags a zone whose occupancy jumps sharply between the last two buckets — a proxy for "something sudden just happened" (most likely a train arrival).

**`src/pipeline.py`** — The orchestrator; the only module that imports from all the others. Notable implementation choices: `detect_every_n_frames` trades detection freshness for speed by reusing the last known boxes between detection passes; `sim_start_time`/`sim_seconds_per_video_second` remap a short demo clip onto a believable rush-hour wall-clock timeline so the trend chart has more than one meaningful data point; the zone-status legend is drawn as a single fixed panel rather than floating text at each zone's centroid, since the latter overlaps badly when zones share a vertical position; video frames are written via `imageio` configured for `codec="libx264", pixelformat="yuv420p"` rather than `cv2.VideoWriter`, because most `opencv-python` builds lack a licensed H.264 encoder and silently fall back to a codec browsers won't play.

**`dashboard/app.py`** — A Streamlit UI with no new logic of its own: it collects sidebar configuration, calls `run_pipeline(...)`, and renders whatever comes back — zone status cards from `summary["final_zone_counts"]`/`final_density_levels`, a Plotly trend chart (with a forecast segment computed via `forecast_next_bucket()`) from `analytics_hourly.csv`, and the heatmap/video/raw-data tabs pointed straight at the files `run_pipeline()` already wrote.

**`tools/zone_selector.py`** — An interactive OpenCV GUI for drawing custom zone polygons on real camera footage instead of the default equal-thirds split: click points, press `n` to name/finish a zone, `s` to save. On save it divides clicked pixel coordinates by frame width/height — the inverse of `scale_to_frame()` — to write zones back out in the same normalized format the rest of the project expects. Needs a local display; won't run on a headless server.

## Key design decisions

- **Foot-point, not box-center**, for zone/ground-position checks — a person's head position drifts more with camera angle and height than their feet do, so bottom-center of the bounding box is a better stand-in for "where someone is standing."
- **Normalized zone coordinates** so one `zones.json` works across any camera resolution without rewriting pixel values.
- **Two detector backends behind one interface** — YOLO is the real path; HOG is a genuine offline/dependency-free fallback, not just a testing shortcut, useful in network-restricted environments.
- **`imageio`/H.264 instead of `cv2.VideoWriter`** for output video, since most `opencv-python` builds can't actually encode browser-playable H.264 despite accepting the codec tag.
- **CWD-independent default paths** in `pipeline.py` — resolved relative to the script's own location, not whatever directory it happens to be invoked from.
