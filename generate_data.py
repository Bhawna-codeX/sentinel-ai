"""
generate_data.py
SentinelAI - Synthetic cybersecurity access-log generator.

Generates a realistic, reproducible synthetic dataset of user / service-account /
device access events with injected behavioral anomalies for 7 attack categories.

Run:
    python generate_data.py
"""

import os
import math
import random
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
N_USERS = 150
N_SERVICE_ACCOUNTS = 20
N_DEVICES_POOL = 30
TOTAL_EVENTS = 22000
ANOMALY_RATE = 0.03  # ~3% anomalous events (within 2-4% range)

START_DATE = datetime(2026, 4, 1)
END_DATE = datetime(2026, 7, 20)
TOTAL_SECONDS = int((END_DATE - START_DATE).total_seconds())

COUNTRIES_CITIES = [
    ("India", "Bhopal"), ("India", "Mumbai"), ("India", "Bengaluru"),
    ("United States", "New York"), ("United States", "San Francisco"),
    ("Germany", "Berlin"), ("United Kingdom", "London"),
    ("Singapore", "Singapore"), ("Japan", "Tokyo"), ("Brazil", "Sao Paulo"),
    ("Russia", "Moscow"), ("Nigeria", "Lagos"), ("Australia", "Sydney"),
    ("Canada", "Toronto"), ("Netherlands", "Amsterdam"),
]

CITY_COORDS = {
    "Bhopal": (23.2599, 77.4126), "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946), "New York": (40.7128, -74.0060),
    "San Francisco": (37.7749, -122.4194), "Berlin": (52.5200, 13.4050),
    "London": (51.5074, -0.1278), "Singapore": (1.3521, 103.8198),
    "Tokyo": (35.6762, 139.6503), "Sao Paulo": (-23.5505, -46.6333),
    "Moscow": (55.7558, 37.6173), "Lagos": (6.5244, 3.3792),
    "Sydney": (-33.8688, 151.2093), "Toronto": (43.6532, -79.3832),
    "Amsterdam": (52.3676, 4.9041),
}

RESOURCES = {
    "hr": ["hr_payroll_db", "hr_employee_records", "hr_benefits_portal"],
    "finance": ["finance_ledger_db", "finance_invoice_sys", "finance_reporting"],
    "engineering": ["source_code_repo", "ci_cd_pipeline", "internal_wiki", "build_server"],
    "customer": ["crm_customer_db", "support_ticket_sys", "billing_portal"],
    "infra": ["cloud_admin_console", "network_config_db", "vpn_gateway", "backup_server"],
    "security": ["siem_console", "identity_admin", "key_vault", "audit_logs"],
}
RESOURCE_CATEGORY = {r: cat for cat, rs in RESOURCES.items() for r in rs}
ALL_RESOURCES = list(RESOURCE_CATEGORY.keys())

AUTH_METHODS = ["password", "mfa_push", "sso", "api_key", "certificate"]
PROTOCOLS = ["HTTPS", "SSH", "RDP", "VPN", "SFTP", "internal_api"]
DEVICE_OS = ["Windows 11", "Windows 10", "macOS", "Ubuntu Linux", "iOS", "Android", "ServerOS"]
PRIVILEGE_LEVELS = ["standard", "elevated", "admin"]

ATTACK_TYPES = [
    "brute_force", "credential_stuffing", "impossible_travel", "lateral_movement",
    "device_spoofing", "low_and_slow_exfiltration", "insider_drift",
]


# --------------------------------------------------------------------------
# Entity baseline generation
# --------------------------------------------------------------------------
def make_entity_pool():
    entities = []
    for i in range(N_USERS):
        entities.append(f"user_{i:04d}")
    for i in range(N_SERVICE_ACCOUNTS):
        entities.append(f"svc_{i:03d}")
    return entities


def entity_type_of(entity_id):
    return "service_account" if entity_id.startswith("svc_") else "user"


def build_baselines(entities):
    """Each entity gets its own normal-behavior profile."""
    baselines = {}
    device_pool = [f"dev_{i:04d}" for i in range(N_DEVICES_POOL)]
    for ent in entities:
        etype = entity_type_of(ent)
        country, city = random.choice(COUNTRIES_CITIES)
        lat, lon = CITY_COORDS[city]
        # service accounts operate mostly off-hours / narrow windows; users 9-6ish
        if etype == "service_account":
            login_hour_center = random.choice([1, 2, 3, 22, 23])
            login_hour_spread = 2
            typical_resources = random.sample(ALL_RESOURCES, k=random.randint(1, 2))
            privilege = random.choice(["elevated", "admin"])
        else:
            login_hour_center = random.randint(8, 17)
            login_hour_spread = 3
            typical_resources = random.sample(ALL_RESOURCES, k=random.randint(2, 4))
            privilege = random.choices(PRIVILEGE_LEVELS, weights=[0.75, 0.2, 0.05])[0]

        baselines[ent] = {
            "entity_type": etype,
            "country": country,
            "city": city,
            "lat": lat,
            "lon": lon,
            "device_id": random.choice(device_pool),
            "device_os": random.choice(DEVICE_OS),
            "device_fingerprint": fake.sha256()[:16],
            "auth_method": random.choices(AUTH_METHODS, weights=[0.3, 0.35, 0.2, 0.1, 0.05])[0],
            "protocol": random.choice(PROTOCOLS),
            "typical_resources": typical_resources,
            "login_hour_center": login_hour_center,
            "login_hour_spread": login_hour_spread,
            "session_duration_mean": np.random.uniform(8, 45),
            "session_duration_std": np.random.uniform(2, 8),
            "bytes_mean": np.random.uniform(50_000, 800_000),
            "bytes_std": np.random.uniform(10_000, 100_000),
            "privilege_level": privilege,
            "command_count_mean": np.random.uniform(3, 20),
        }
    return baselines


def random_timestamp():
    offset = random.randint(0, TOTAL_SECONDS)
    return START_DATE + timedelta(seconds=offset)


def normal_event(entity_id, baseline, event_counter):
    """Generate one plausible normal event for an entity, with noise."""
    ts = random_timestamp()
    hour = int(np.clip(np.random.normal(baseline["login_hour_center"], baseline["login_hour_spread"]), 0, 23))
    ts = ts.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))

    resource = random.choice(baseline["typical_resources"])
    session_duration = max(1, np.random.normal(baseline["session_duration_mean"], baseline["session_duration_std"]))
    bytes_transferred = max(1000, np.random.normal(baseline["bytes_mean"], baseline["bytes_std"]))
    country, city, lat, lon = baseline["country"], baseline["city"], baseline["lat"], baseline["lon"]

    row = {
        "event_id": event_counter,
        "entity_id": entity_id,
        "entity_type": baseline["entity_type"],
        "timestamp": ts,
        "source_ip": fake.ipv4_public(),
        "country": country,
        "city": city,
        "latitude": lat + np.random.normal(0, 0.01),
        "longitude": lon + np.random.normal(0, 0.01),
        "resource_accessed": resource,
        "resource_category": RESOURCE_CATEGORY[resource],
        "auth_method": baseline["auth_method"],
        "login_status": "success",
        "failed_attempts_last_10_min": np.random.poisson(0.1),
        "session_duration_minutes": round(session_duration, 2),
        "device_id": baseline["device_id"],
        "device_os": baseline["device_os"],
        "device_fingerprint": baseline["device_fingerprint"],
        "protocol": baseline["protocol"],
        "bytes_transferred": round(bytes_transferred, 2),
        "privilege_level": baseline["privilege_level"],
        "command_count": max(0, int(np.random.normal(baseline["command_count_mean"], 3))),
        "new_device": 0,
        "new_location": 0,
        "geo_velocity_kmph": round(abs(np.random.normal(0, 3)), 2),
        "anomaly_label": 0,
        "attack_type": "normal",
    }
    return row


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def attack_event(entity_id, baseline, event_counter, attack_type):
    """Craft an anomalous event according to the attack pattern."""
    row = normal_event(entity_id, baseline, event_counter)
    row["anomaly_label"] = 1
    row["attack_type"] = attack_type

    if attack_type == "brute_force":
        row["failed_attempts_last_10_min"] = random.randint(8, 30)
        row["login_status"] = random.choice(["failed", "failed", "success"])
        row["session_duration_minutes"] = round(np.random.uniform(0.1, 2), 2)

    elif attack_type == "credential_stuffing":
        row["source_ip"] = fake.ipv4_public()
        row["failed_attempts_last_10_min"] = random.randint(5, 25)
        row["login_status"] = random.choice(["failed", "success"])
        row["auth_method"] = "password"
        row["session_duration_minutes"] = round(np.random.uniform(0.1, 1.5), 2)

    elif attack_type == "impossible_travel":
        _, far_city = random.choice([c for c in COUNTRIES_CITIES if c[1] != baseline["city"]])
        flat, flon = CITY_COORDS[far_city]
        dist = haversine_km(baseline["lat"], baseline["lon"], flat, flon)
        dt_minutes = np.random.uniform(2, 20)
        row["latitude"], row["longitude"] = flat, flon
        row["country"] = [c for c, ct in COUNTRIES_CITIES if ct == far_city][0]
        row["city"] = far_city
        row["geo_velocity_kmph"] = round(dist / (dt_minutes / 60), 2)
        row["new_location"] = 1

    elif attack_type == "lateral_movement":
        n_resources = random.randint(4, 8)
        row["resource_accessed"] = random.choice(ALL_RESOURCES)
        row["resource_category"] = RESOURCE_CATEGORY[row["resource_accessed"]]
        row["command_count"] = int(baseline["command_count_mean"] * random.uniform(3, 6))
        row["privilege_level"] = random.choice(["elevated", "admin"])
        row["_unique_resources_hint"] = n_resources

    elif attack_type == "device_spoofing":
        row["device_id"] = baseline["device_id"]  # same known device id
        row["device_fingerprint"] = fake.sha256()[:16]  # mismatched fingerprint
        row["device_os"] = random.choice([o for o in DEVICE_OS if o != baseline["device_os"]])
        row["protocol"] = random.choice([p for p in PROTOCOLS if p != baseline["protocol"]])
        row["new_device"] = 1

    elif attack_type == "low_and_slow_exfiltration":
        row["hour_override"] = random.choice([1, 2, 3, 4])
        row["bytes_transferred"] = round(baseline["bytes_mean"] * random.uniform(1.5, 3.0), 2)
        sensitive = [r for r in ALL_RESOURCES if RESOURCE_CATEGORY[r] in ("finance", "security", "hr")]
        row["resource_accessed"] = random.choice(sensitive)
        row["resource_category"] = RESOURCE_CATEGORY[row["resource_accessed"]]
        row["session_duration_minutes"] = round(np.random.uniform(30, 90), 2)

    elif attack_type == "insider_drift":
        row["privilege_level"] = random.choice(["elevated", "admin"])
        row["resource_accessed"] = random.choice(ALL_RESOURCES)
        row["resource_category"] = RESOURCE_CATEGORY[row["resource_accessed"]]
        row["command_count"] = int(baseline["command_count_mean"] * random.uniform(1.5, 2.5))

    if "hour_override" in row:
        row["timestamp"] = row["timestamp"].replace(hour=row.pop("hour_override"))

    return row


def generate_dataset():
    entities = make_entity_pool()
    baselines = build_baselines(entities)

    n_anomalies = int(TOTAL_EVENTS * ANOMALY_RATE)
    n_normal = TOTAL_EVENTS - n_anomalies

    rows = []
    counter = 1

    # Normal events, weighted so entities with more history look realistic
    for _ in range(n_normal):
        ent = random.choice(entities)
        rows.append(normal_event(ent, baselines[ent], counter))
        counter += 1

    # Anomalous events spread across attack types
    per_type = n_anomalies // len(ATTACK_TYPES)
    remainder = n_anomalies - per_type * len(ATTACK_TYPES)
    for i, atype in enumerate(ATTACK_TYPES):
        n = per_type + (1 if i < remainder else 0)
        for _ in range(n):
            ent = random.choice(entities)
            rows.append(attack_event(ent, baselines[ent], counter, atype))
            counter += 1

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["event_id"] = range(1, len(df) + 1)

    # Derived temporal columns
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Rolling / sequential features computed per-entity in chronological order
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    time_since = []
    unique_res_24h = []
    entity_history_len = []
    seen_resources = {}
    last_ts = {}
    history_count = {}

    for idx, r in df.iterrows():
        ent = r["entity_id"]
        ts = r["timestamp"]

        if ent in last_ts:
            delta_min = (ts - last_ts[ent]).total_seconds() / 60
        else:
            delta_min = np.nan
        time_since.append(delta_min)
        last_ts[ent] = ts

        seen_resources.setdefault(ent, set())
        seen_resources[ent].add(r["resource_accessed"])
        unique_res_24h.append(len(seen_resources[ent]))

        history_count[ent] = history_count.get(ent, 0) + 1
        entity_history_len.append(history_count[ent])

    df["time_since_last_event_minutes"] = time_since
    df["unique_resources_last_24h"] = unique_res_24h
    df["entity_history_length"] = entity_history_len

    # apply lateral-movement resource-count hint if present
    if "_unique_resources_hint" in df.columns:
        mask = df["_unique_resources_hint"].notna()
        df.loc[mask, "unique_resources_last_24h"] = (
            df.loc[mask, "unique_resources_last_24h"] + df.loc[mask, "_unique_resources_hint"]
        )
        df = df.drop(columns=["_unique_resources_hint"])

    df["time_since_last_event_minutes"] = df["time_since_last_event_minutes"].fillna(
        df["time_since_last_event_minutes"].median()
    )

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["event_id"] = range(1, len(df) + 1)

    ordered_cols = [
        "event_id", "entity_id", "entity_type", "timestamp", "source_ip", "country", "city",
        "latitude", "longitude", "resource_accessed", "resource_category", "auth_method",
        "login_status", "failed_attempts_last_10_min", "session_duration_minutes", "hour",
        "day_of_week", "is_weekend", "device_id", "device_os", "device_fingerprint", "protocol",
        "bytes_transferred", "privilege_level", "command_count", "unique_resources_last_24h",
        "new_device", "new_location", "geo_velocity_kmph", "time_since_last_event_minutes",
        "entity_history_length", "anomaly_label", "attack_type",
    ]
    df = df[ordered_cols]
    return df, baselines


def save_baselines(baselines):
    """Persist entity baselines for use by the dashboard (entity profile page)."""
    import json
    out = {}
    for ent, b in baselines.items():
        out[ent] = {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in b.items()}
    with open(os.path.join(DATA_DIR, "entity_baselines.json"), "w") as f:
        json.dump(out, f, indent=2)


def chronological_split(df, train_frac=0.75):
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_frac)
    train_df = df_sorted.iloc[:split_idx].reset_index(drop=True)
    test_df = df_sorted.iloc[split_idx:].reset_index(drop=True)
    return train_df, test_df


def main():
    print("Generating synthetic access log dataset ...")
    df, baselines = generate_dataset()
    print(f"Total events generated: {len(df)}")
    print(df["attack_type"].value_counts())

    full_path = os.path.join(DATA_DIR, "synthetic_access_logs.csv")
    df.to_csv(full_path, index=False)
    print(f"Saved: {full_path}")

    train_df, test_df = chronological_split(df)
    train_df.to_csv(os.path.join(DATA_DIR, "train_data.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test_data.csv"), index=False)
    print(f"Saved chronological train/test split: {len(train_df)} train / {len(test_df)} test")

    save_baselines(baselines)
    print("Saved entity baselines JSON.")
    print("Data generation complete.")


if __name__ == "__main__":
    main()
