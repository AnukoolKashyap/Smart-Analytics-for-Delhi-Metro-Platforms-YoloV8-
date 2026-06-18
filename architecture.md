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
                         │   (ZoneManager)      │    each person standing in?"
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