"""
utils.py
SentinelAI - shared helpers: feature engineering, baseline profiling, paths.
"""

import os
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

for d in (DATA_DIR, MODEL_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

RAW_CSV = os.path.join(DATA_DIR, "synthetic_access_logs.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "train_data.csv")
TEST_CSV = os.path.join(DATA_DIR, "test_data.csv")
BASELINE_JSON = os.path.join(DATA_DIR, "entity_baselines.json")

CATEGORICAL_COLS = [
    "entity_type", "country", "city", "resource_accessed", "resource_category",
    "auth_method", "login_status", "device_os", "protocol", "privilege_level",
]

FEATURE_COLUMNS = [
    "login_hour_deviation", "session_duration_deviation", "resource_novelty_score",
    "device_novelty_score", "location_novelty_score", "auth_novelty_score",
    "protocol_novelty_score", "failed_login_velocity", "unique_resources_last_24h",
    "bytes_transfer_deviation", "privilege_deviation", "geo_velocity_kmph",
    "time_since_last_event_minutes", "new_device", "new_location", "is_off_hours",
    "is_weekend", "entity_history_length", "cold_start_indicator",
    "entity_type_enc", "country_enc", "resource_category_enc", "auth_method_enc",
    "protocol_enc", "privilege_level_enc",
]

PRIVILEGE_RANK = {"standard": 0, "elevated": 1, "admin": 2}
COLD_START_THRESHOLD = 5  # events; below this an entity is "cold start"


def load_baselines():
    if not os.path.exists(BASELINE_JSON):
        return {}
    with open(BASELINE_JSON, "r") as f:
        return json.load(f)


def _safe_encode(series, mapping):
    return series.map(mapping).fillna(-1).astype(int)


def build_label_encoders(df):
    """Fit simple frequency-based label encoders for categorical columns
    (robust to unseen categories at inference: they map to -1)."""
    encoders = {}
    for col in CATEGORICAL_COLS:
        cats = sorted(df[col].dropna().unique().tolist())
        encoders[col] = {c: i for i, c in enumerate(cats)}
    return encoders


def apply_encoders(df, encoders):
    out = df.copy()
    for col, mapping in encoders.items():
        out[f"{col}_enc"] = _safe_encode(out[col], mapping)
    return out


def engineer_features(df, baselines=None, entity_history=None):
    """
    Compute behavioral features per event using each entity's baseline profile.
    If `baselines` is None, per-dataset empirical baselines are derived on the fly
    (used for training). For live/simulated single-event scoring, pass the
    precomputed baselines dict (from entity_baselines.json) plus optional
    entity_history (dict of counts) to correctly flag cold-start entities.

    Label leakage prevention: anomaly_label / attack_type are NEVER read here.
    """
    df = df.copy()

    if baselines is None:
        # Derive empirical per-entity baselines from the given dataframe itself
        # (used consistently for both train and test to avoid leakage from labels,
        # since only feature columns and entity_id/timestamp are used).
        grp = df.groupby("entity_id")
        baseline_hour = grp["hour"].transform("mean")
        baseline_hour_std = grp["hour"].transform("std").fillna(3).replace(0, 3)
        baseline_session = grp["session_duration_minutes"].transform("mean")
        baseline_session_std = grp["session_duration_minutes"].transform("std").fillna(5).replace(0, 5)
        baseline_bytes = grp["bytes_transferred"].transform("mean")
        baseline_bytes_std = grp["bytes_transferred"].transform("std").fillna(50000).replace(0, 50000)
        entity_hist_len = grp.cumcount() + 1

        # Most common (mode) categorical values per entity, precomputed once
        mode_device = grp["device_id"].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        mode_city = grp["city"].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        mode_auth = grp["auth_method"].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        mode_protocol = grp["protocol"].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        mode_resource_cat = grp["resource_category"].transform(
            lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]
        )
        mode_privilege = grp["privilege_level"].transform(
            lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]
        )

        df["login_hour_deviation"] = (df["hour"] - baseline_hour).abs() / baseline_hour_std
        df["session_duration_deviation"] = (
            (df["session_duration_minutes"] - baseline_session).abs() / baseline_session_std
        )
        df["bytes_transfer_deviation"] = (
            (df["bytes_transferred"] - baseline_bytes).abs() / baseline_bytes_std
        )
        df["device_novelty_score"] = (df["device_id"] != mode_device).astype(float)
        df["location_novelty_score"] = (df["city"] != mode_city).astype(float)
        df["auth_novelty_score"] = (df["auth_method"] != mode_auth).astype(float)
        df["protocol_novelty_score"] = (df["protocol"] != mode_protocol).astype(float)
        df["resource_novelty_score"] = (df["resource_category"] != mode_resource_cat).astype(float)
        df["privilege_deviation"] = (
            df["privilege_level"].map(PRIVILEGE_RANK).fillna(0)
            - mode_privilege.map(PRIVILEGE_RANK).fillna(0)
        ).clip(lower=0)
        df["entity_history_length"] = entity_hist_len
        df["cold_start_indicator"] = (entity_hist_len <= COLD_START_THRESHOLD).astype(int)

    else:
        # Single-event / streaming scoring path using precomputed baselines
        rows = []
        for idx, r in df.iterrows():
            b = baselines.get(r["entity_id"])
            hist_len = (entity_history or {}).get(r["entity_id"], 0) + 1
            if b is None:
                rows.append(dict(
                    login_hour_deviation=1.0, session_duration_deviation=1.0,
                    bytes_transfer_deviation=1.0, device_novelty_score=1.0,
                    location_novelty_score=1.0, auth_novelty_score=1.0,
                    protocol_novelty_score=1.0, resource_novelty_score=1.0,
                    privilege_deviation=1.0, entity_history_length=hist_len,
                    cold_start_indicator=1,
                ))
                continue
            hour_std = b.get("login_hour_spread", 3) or 3
            session_std = b.get("session_duration_std", 5) or 5
            bytes_std = b.get("bytes_std", 50000) or 50000
            rows.append(dict(
                login_hour_deviation=abs(r["hour"] - b["login_hour_center"]) / hour_std,
                session_duration_deviation=abs(r["session_duration_minutes"] - b["session_duration_mean"]) / session_std,
                bytes_transfer_deviation=abs(r["bytes_transferred"] - b["bytes_mean"]) / bytes_std,
                device_novelty_score=float(r["device_id"] != b["device_id"]),
                location_novelty_score=float(r["city"] != b["city"]),
                auth_novelty_score=float(r["auth_method"] != b["auth_method"]),
                protocol_novelty_score=float(r["protocol"] != b["protocol"]),
                resource_novelty_score=float(r["resource_accessed"] not in b.get("typical_resources", [])),
                privilege_deviation=max(0, PRIVILEGE_RANK.get(r["privilege_level"], 0)
                                         - PRIVILEGE_RANK.get(b["privilege_level"], 0)),
                entity_history_length=hist_len,
                cold_start_indicator=int(hist_len <= COLD_START_THRESHOLD),
            ))
        feat_df = pd.DataFrame(rows, index=df.index)
        for c in feat_df.columns:
            df[c] = feat_df[c]

    df["failed_login_velocity"] = df["failed_attempts_last_10_min"].astype(float)
    df["is_off_hours"] = ((df["hour"] < 6) | (df["hour"] > 21)).astype(int)
    df["is_weekend"] = df["is_weekend"].astype(int)
    df["unique_resources_last_24h"] = df["unique_resources_last_24h"].astype(float)
    df["geo_velocity_kmph"] = df["geo_velocity_kmph"].astype(float)
    df["time_since_last_event_minutes"] = df["time_since_last_event_minutes"].fillna(
        df["time_since_last_event_minutes"].median() if df["time_since_last_event_minutes"].notna().any() else 60.0
    )
    df["new_device"] = df["new_device"].astype(int) if "new_device" in df else 0
    df["new_location"] = df["new_location"].astype(int) if "new_location" in df else 0

    # replace inf/-inf and clip extreme deviations for numerical stability
    dev_cols = ["login_hour_deviation", "session_duration_deviation", "bytes_transfer_deviation"]
    for c in dev_cols:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 20)

    return df


def get_feature_matrix(df, encoders):
    """Return the final numeric feature matrix (X) using FEATURE_COLUMNS."""
    df_enc = apply_encoders(df, encoders)
    missing = [c for c in FEATURE_COLUMNS if c not in df_enc.columns]
    for c in missing:
        df_enc[c] = 0
    X = df_enc[FEATURE_COLUMNS].fillna(0)
    return X


def risk_level_from_score(score):
    if score >= 80:
        return "Critical"
    elif score >= 60:
        return "High"
    elif score >= 30:
        return "Medium"
    return "Low"
