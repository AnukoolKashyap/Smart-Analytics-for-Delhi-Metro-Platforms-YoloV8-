"""
dashboard/app.py
-----------------
Streamlit front-end for the crowd analytics pipeline.

Run with:
    streamlit run dashboard/app.py
"""

import json
import os
import sys
import tempfile
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from analytics import forecast_next_bucket  # noqa: E402
from pipeline import run_pipeline  # noqa: E402
from zones import ZoneManager  # noqa: E402

DEFAULT_ZONES_PATH = os.path.join(PROJECT_ROOT, "config", "zones.json")
SAMPLE_VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "sample_platform_footage.avi")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

LEVEL_COLOR = {"LOW": "#22c55e", "MEDIUM": "#eab308", "HIGH": "#f97316", "CRITICAL": "#ef4444"}

st.set_page_config(page_title="Delhi Metro Crowd Analytics", layout="wide", page_icon="🚇")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    .zone-card {
        border-radius: 10px; padding: 16px 18px; background: #161b22;
        border: 1px solid #2a3038;
    }
    .zone-card h4 {margin: 0 0 6px 0; font-size: 0.95rem; color: #9aa4af; font-weight: 600;}
    .zone-count {font-size: 2rem; font-weight: 700; margin: 0;}
    .zone-badge {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.75rem; font-weight: 700; color: #0b0e12; margin-top: 6px;
    }
    .alert-box {
        border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;
        border-left: 4px solid; font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def hhmm(total_seconds: float) -> str:
    total_seconds = int(total_seconds) % 86400
    return (timedelta(seconds=total_seconds).__str__())[:-3] if total_seconds % 60 == 0 else timedelta(seconds=total_seconds).__str__()


def zone_card(name: str, count: int, capacity: int, level: str) -> str:
    color = LEVEL_COLOR[level]
    return f"""
    <div class="zone-card">
        <h4>{name}</h4>
        <p class="zone-count">{count} <span style="font-size:1rem;color:#7a8794;">/ {capacity}</span></p>
        <span class="zone-badge" style="background:{color};">{level}</span>
    </div>
    """


st.title("🚇 Delhi Metro Smart Crowd Analytics")
st.caption(
    "YOLO-based platform crowd monitoring — zone-wise occupancy, hourly trend analytics, "
    "density heatmaps, and redistribution alerts. Personal project / prototype, not affiliated with DMRC."
)

# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("Configuration")

    video_choice = st.radio("Video source", ["Bundled sample footage", "Upload my own video"])
    if video_choice == "Upload my own video":
        uploaded = st.file_uploader("Upload a platform/crowd video", type=["mp4", "avi", "mov", "mkv"])
        video_path = None
        if uploaded is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
            tmp.write(uploaded.read())
            tmp.close()
            video_path = tmp.name
    else:
        video_path = SAMPLE_VIDEO_PATH
        st.caption(
            "Bundled clip is a generic pedestrian test video used as a placeholder — "
            "swap in real platform/CCTV-style footage via upload for an actual metro demo."
        )

    backend = st.selectbox(
        "Detector backend", ["yolo", "hog"],
        help=(
            "yolo: YOLOv8 + ByteTrack (recommended). Downloads model weights on first run, "
            "so needs internet once. hog: fully offline OpenCV fallback, lower accuracy — "
            "useful for a quick test with no internet access."
        ),
    )

    sim_start = st.text_input("Simulated rush-hour start (HH:MM)", value="08:00")
    sim_scale = st.slider(
        "Simulated minutes per video-second", 1, 120, 60,
        help="Stretches short demo footage onto a realistic rush-hour timeline for the trend chart.",
    )

    bucket_label = st.selectbox("Aggregation bucket", ["5 min", "10 min", "30 min", "1 hour"], index=1)
    bucket_seconds = {"5 min": 300, "10 min": 600, "30 min": 1800, "1 hour": 3600}[bucket_label]

    max_frames = st.slider("Max frames to process (speed cap)", 50, 800, 350, step=50)
    detect_every = st.slider("Run detector every N frames", 1, 5, 2)

    run_clicked = st.button("▶ Run Analysis", type="primary", use_container_width=True)

# ------------------------------------------------------------- run logic --
if run_clicked:
    if video_path is None:
        st.error("Upload a video first, or switch to the bundled sample footage.")
    else:
        with st.spinner("Running detection, zone assignment, heatmap accumulation, and analytics..."):
            try:
                summary = run_pipeline(
                    video_path=video_path,
                    zones_config_path=DEFAULT_ZONES_PATH,
                    output_dir=OUTPUT_DIR,
                    backend=backend,
                    detect_every_n_frames=detect_every,
                    bucket_seconds=bucket_seconds,
                    sim_start_time=sim_start if len(sim_start.split(":")) > 1 else "08:00",
                    sim_seconds_per_video_second=float(sim_scale),
                    max_frames=max_frames,
                )
                st.session_state["summary"] = summary
            except Exception as e:
                st.error(f"Pipeline failed: {e}")

# ------------------------------------------------------------- results ---
if "summary" in st.session_state:
    summary = st.session_state["summary"]
    zone_mgr = ZoneManager(DEFAULT_ZONES_PATH)
    zone_names = {z.id: z.name for z in zone_mgr.zones}
    capacities = {z.id: z.capacity for z in zone_mgr.zones}
    zone_ids = list(zone_names.keys())

    st.markdown("### Live zone status (last processed frame)")
    cols = st.columns(len(zone_ids))
    for col, zid in zip(cols, zone_ids):
        count = summary["final_zone_counts"].get(zid, 0)
        level = summary["final_density_levels"].get(zid, "LOW")
        col.markdown(zone_card(zone_names[zid], count, capacities[zid], level), unsafe_allow_html=True)

    st.markdown("####")
    alert_col, meta_col = st.columns([2, 1])
    with alert_col:
        if summary.get("redistribution_suggestion"):
            st.markdown(
                f'<div class="alert-box" style="border-color:#3b82f6;background:#0f1b2e;">'
                f'🧭 <b>Redistribution suggestion:</b> {summary["redistribution_suggestion"]}</div>',
                unsafe_allow_html=True,
            )
        for s in summary.get("surge_alerts", []):
            st.markdown(
                f'<div class="alert-box" style="border-color:#ef4444;background:#2a1212;">'
                f'⚠️ <b>Surge alert — {s["zone_name"]}:</b> occupancy jumped '
                f'{s["pct_increase"]:.0f}% bucket-over-bucket ({s["previous"]:.1f} → {s["current"]:.1f} avg)</div>',
                unsafe_allow_html=True,
            )
        if not summary.get("redistribution_suggestion") and not summary.get("surge_alerts"):
            st.markdown(
                '<div class="alert-box" style="border-color:#22c55e;background:#0f1f14;">'
                "✅ No redistribution or surge alerts for this run — load looks balanced.</div>",
                unsafe_allow_html=True,
            )
    with meta_col:
        st.metric("Frames processed", summary["frames_processed"])
        st.metric("Processing speed", f'{summary["fps_processing_speed"]} fps')

    tab_trends, tab_heatmap, tab_video, tab_data = st.tabs(
        ["📈 Hourly Trends & Forecast", "🔥 Crowd Heatmap", "🎥 Annotated Video", "📄 Raw Data"]
    )

    hourly_df = pd.read_csv(summary["outputs"]["analytics_hourly_csv"])

    with tab_trends:
        if hourly_df.empty:
            st.info("Not enough data collected to build a trend chart — try a longer clip or smaller bucket size.")
        else:
            fig = go.Figure()
            time_labels = [hhmm(s) for s in hourly_df["bucket_start_sec"]]
            for zid in zone_ids:
                col = f"{zid}_mean"
                if col not in hourly_df.columns:
                    continue
                means = hourly_df[col].tolist()
                fig.add_trace(go.Scatter(x=time_labels, y=means, mode="lines+markers", name=zone_names[zid]))

                forecast_val = forecast_next_bucket(means)
                if forecast_val is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=[time_labels[-1], "next"], y=[means[-1], forecast_val],
                            mode="lines+markers", line=dict(dash="dot"),
                            marker=dict(symbol="diamond"), name=f"{zone_names[zid]} (forecast)",
                            showlegend=True,
                        )
                    )
            fig.update_layout(
                template="plotly_dark", height=440,
                xaxis_title="Time", yaxis_title="Avg. people in zone",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Dotted segment is a naive linear-trend forecast for the next bucket, "
                "not a trained time-series model — directionally useful, not a precise prediction."
            )

    with tab_heatmap:
        if os.path.exists(summary["outputs"]["heatmap"]):
            st.image(summary["outputs"]["heatmap"], caption="Accumulated foot-traffic density over the whole clip")
        else:
            st.info("Heatmap not generated yet.")

    with tab_video:
        if os.path.exists(summary["outputs"]["annotated_video"]):
            st.video(summary["outputs"]["annotated_video"])
        else:
            st.info("Annotated video not generated yet.")

    with tab_data:
        raw_df = pd.read_csv(summary["outputs"]["analytics_raw_csv"])
        st.markdown("**Raw per-detection-pass log (tail)**")
        st.dataframe(raw_df.tail(30), use_container_width=True)
        st.markdown("**Bucketed (hourly-style) aggregation**")
        st.dataframe(hourly_df, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.download_button("Download raw CSV", raw_df.to_csv(index=False), file_name="analytics_raw.csv")
        c2.download_button("Download bucketed CSV", hourly_df.to_csv(index=False), file_name="analytics_hourly.csv")
else:
    st.info("Configure options in the sidebar and click **Run Analysis** to process footage.")
