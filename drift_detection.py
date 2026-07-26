"""
drift_detection.py
SentinelAI - lightweight concept-drift detection using Population Stability
Index (PSI) between a historical baseline window and a recent window.

Also implements cold-start entity detection and a trusted-profile-update
policy description used by the dashboard.
"""

import numpy as np
import pandas as pd

DRIFT_FEATURES = [
    "login_hour_deviation", "session_duration_deviation", "bytes_transfer_deviation",
    "failed_login_velocity", "geo_velocity_kmph", "unique_resources_last_24h",
]


def _psi_for_feature(baseline, recent, buckets=10):
    baseline = np.asarray(baseline, dtype=float)
    recent = np.asarray(recent, dtype=float)
    baseline = baseline[~np.isnan(baseline)]
    recent = recent[~np.isnan(recent)]
    if len(baseline) < 10 or len(recent) < 10:
        return 0.0

    edges = np.unique(np.quantile(baseline, np.linspace(0, 1, buckets + 1)))
    if len(edges) < 3:
        return 0.0

    base_counts, _ = np.histogram(baseline, bins=edges)
    recent_counts, _ = np.histogram(recent, bins=edges)

    base_pct = np.clip(base_counts / max(base_counts.sum(), 1), 1e-4, None)
    recent_pct = np.clip(recent_counts / max(recent_counts.sum(), 1), 1e-4, None)

    psi = np.sum((recent_pct - base_pct) * np.log(recent_pct / base_pct))
    return float(psi)


def compute_drift_report(df, feature_cols=None, split_frac=0.5):
    """Split chronologically into baseline (older) vs recent window and
    compute PSI per feature. PSI > 0.2 = moderate drift, > 0.25 = significant."""
    feature_cols = feature_cols or DRIFT_FEATURES
    df_sorted = df.sort_values("timestamp") if "timestamp" in df.columns else df
    split_idx = int(len(df_sorted) * split_frac)
    baseline_df = df_sorted.iloc[:split_idx]
    recent_df = df_sorted.iloc[split_idx:]

    results = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        psi = _psi_for_feature(baseline_df[col], recent_df[col])
        if psi < 0.1:
            status = "Stable"
        elif psi < 0.25:
            status = "Moderate Drift"
        else:
            status = "Significant Drift"
        results.append({"feature": col, "psi": round(psi, 4), "status": status})

    return pd.DataFrame(results).sort_values("psi", ascending=False).reset_index(drop=True)


def cold_start_entities(df, threshold=5):
    counts = df.groupby("entity_id").size().reset_index(name="event_count")
    counts["cold_start"] = counts["event_count"] <= threshold
    return counts.sort_values("event_count")


PROFILE_UPDATE_POLICY = (
    "Entity behavioral baselines are only updated after a trusted observation "
    "window of at least 30 consecutive low-risk events for that entity. "
    "Sessions flagged as Critical or High risk are never used to update a "
    "baseline automatically - this prevents an attacker's behavior from being "
    "silently 'learned' as normal. Profile updates are logged and reversible."
)

TRUSTED_WINDOW_EVENTS = 30
