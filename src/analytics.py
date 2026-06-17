"""
analytics.py
------------
Everything downstream of "how many people are in each zone right now":

  AnalyticsLogger        per-timestamp record keeping + CSV/JSON export
  bucket_aggregate()     groups the raw log into fixed-size time buckets
                         (use bucket_seconds=3600 on real footage for true
                         hourly analytics; demo clips use a smaller bucket
                         so the trend chart has more than one data point)
  redistribution_suggestion()   compares zone occupancy ratios and proposes
                         where to redirect incoming passengers
  forecast_next_bucket()  naive short-horizon forecast (linear fit over the
                         last few buckets) per zone -- enough to show a
                         "predicted next interval" without needing a real
                         time-series model for a 2-day build
  detect_surge()         flags a zone whose count jumped sharply between
                         consecutive buckets (possible stampede-risk signal)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class AnalyticsLogger:
    def __init__(self):
        self._rows: List[dict] = []

    def record(self, timestamp_sec: float, zone_counts: Dict[str, int]) -> None:
        row = {"timestamp_sec": timestamp_sec}
        row.update(zone_counts)
        self._rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=["timestamp_sec"])
        return pd.DataFrame(self._rows)

    def export_csv(self, path: str) -> None:
        self.to_dataframe().to_csv(path, index=False)

    def export_json(self, path: str) -> None:
        self.to_dataframe().to_json(path, orient="records")


def bucket_aggregate(df: pd.DataFrame, zone_ids: List[str], bucket_seconds: float) -> pd.DataFrame:
    """Group raw per-frame rows into fixed time buckets, taking the mean
    and max occupancy of each zone per bucket."""
    if df.empty:
        return pd.DataFrame(columns=["bucket_start_sec"] + zone_ids)

    work = df.copy()
    work["bucket"] = (work["timestamp_sec"] // bucket_seconds).astype(int)

    agg_funcs = {z: ["mean", "max"] for z in zone_ids if z in work.columns}
    grouped = work.groupby("bucket").agg(agg_funcs)
    grouped.columns = [f"{z}_{stat}" for z, stat in grouped.columns]
    grouped = grouped.reset_index()
    grouped["bucket_start_sec"] = grouped["bucket"] * bucket_seconds
    grouped = grouped.drop(columns=["bucket"])
    return grouped


def redistribution_suggestion(
    zone_counts: Dict[str, int], capacities: Dict[str, int], zone_names: Dict[str, str],
    min_gap_ratio: float = 0.35,
) -> Optional[str]:
    """If one zone is meaningfully more crowded (relative to its own
    capacity) than another, suggest redirecting incoming passengers."""
    if len(zone_counts) < 2:
        return None

    ratios = {
        zid: (zone_counts.get(zid, 0) / capacities[zid] if capacities.get(zid) else 0)
        for zid in zone_counts
    }
    fullest = max(ratios, key=ratios.get)
    emptiest = min(ratios, key=ratios.get)
    gap = ratios[fullest] - ratios[emptiest]

    if fullest == emptiest or gap < min_gap_ratio:
        return None

    return (
        f"{zone_names.get(fullest, fullest)} is at {ratios[fullest]*100:.0f}% of capacity while "
        f"{zone_names.get(emptiest, emptiest)} is at {ratios[emptiest]*100:.0f}% -- "
        f"direct incoming passengers toward {zone_names.get(emptiest, emptiest)}."
    )


def forecast_next_bucket(bucket_means: List[float]) -> Optional[float]:
    """Naive linear-trend forecast for the next bucket from the last few
    bucket means. Returns None if there isn't enough history yet."""
    history = [v for v in bucket_means if v is not None and not np.isnan(v)]
    if len(history) < 2:
        return None
    recent = history[-4:]  # last up-to-4 buckets keeps it responsive to recent trend
    x = np.arange(len(recent))
    slope, intercept = np.polyfit(x, recent, 1)
    next_val = slope * len(recent) + intercept
    return max(0.0, float(next_val))


@dataclass
class SurgeAlert:
    zone_id: str
    zone_name: str
    previous: float
    current: float
    pct_increase: float


def detect_surge(
    bucket_df: pd.DataFrame, zone_ids: List[str], zone_names: Dict[str, str], threshold_pct: float = 50.0
) -> List[SurgeAlert]:
    """Compare the last two buckets for each zone; flag a sudden jump.
    threshold_pct=50 means a >50% jump bucket-over-bucket trips the alert."""
    alerts = []
    if len(bucket_df) < 2:
        return alerts

    last_two = bucket_df.tail(2)
    for zid in zone_ids:
        col = f"{zid}_mean"
        if col not in last_two.columns:
            continue
        prev, curr = last_two[col].iloc[0], last_two[col].iloc[1]
        if prev <= 0:
            continue
        pct = (curr - prev) / prev * 100
        if pct >= threshold_pct:
            alerts.append(SurgeAlert(zid, zone_names.get(zid, zid), prev, curr, pct))
    return alerts
