"""
risk_engine.py
SentinelAI - deterministic, explainable 0-100 risk scoring engine.

The score is a transparent weighted combination of normalized behavioral
signals plus the ML anomaly score and attack-classifier confidence. It is
intentionally rule-based (rather than a black box) so every alert can be
justified to a SOC analyst.
"""

import numpy as np

# Weights sum to 1.0. Chosen heuristically for hackathon interpretability,
# not statistically optimized - documented clearly as a prototype design choice.
WEIGHTS = {
    "anomaly_score": 0.20,
    "failed_login_velocity": 0.12,
    "geo_velocity": 0.12,
    "device_novelty": 0.10,
    "location_novelty": 0.08,
    "resource_novelty": 0.08,
    "privilege_deviation": 0.10,
    "off_hours": 0.06,
    "bytes_deviation": 0.08,
    "classifier_confidence": 0.06,
}


def _norm(value, cap):
    return float(np.clip(value / cap, 0, 1))


def compute_risk_score(row, anomaly_score_norm, classifier_confidence):
    """
    row: dict-like with engineered feature values for a single event.
    anomaly_score_norm: float in [0,1], higher = more anomalous (already normalized).
    classifier_confidence: float in [0,1], confidence of the predicted attack type
                            (ignored / down-weighted for 'normal' predictions).
    Returns: (risk_score:int[0-100], components:dict of contribution scores 0-1)
    """
    components = {
        "anomaly_score": float(np.clip(anomaly_score_norm, 0, 1)),
        "failed_login_velocity": _norm(row.get("failed_login_velocity", 0), 15),
        "geo_velocity": _norm(row.get("geo_velocity_kmph", 0), 2000),
        "device_novelty": float(np.clip(row.get("device_novelty_score", 0), 0, 1)),
        "location_novelty": float(np.clip(row.get("location_novelty_score", 0), 0, 1)),
        "resource_novelty": float(np.clip(row.get("resource_novelty_score", 0), 0, 1)),
        "privilege_deviation": _norm(row.get("privilege_deviation", 0), 2),
        "off_hours": float(np.clip(row.get("is_off_hours", 0), 0, 1)),
        "bytes_deviation": _norm(row.get("bytes_transfer_deviation", 0), 5),
        "classifier_confidence": float(np.clip(classifier_confidence, 0, 1)),
    }

    weighted_sum = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
    score = int(round(weighted_sum * 100))
    score = int(np.clip(score, 0, 100))
    return score, components


def risk_level(score):
    if score >= 80:
        return "Critical"
    elif score >= 60:
        return "High"
    elif score >= 30:
        return "Medium"
    return "Low"


def recommended_action(attack_type, risk_lvl):
    actions = {
        "brute_force": "Temporarily block source IP and enforce step-up authentication.",
        "credential_stuffing": "Block offending source IP(s) and force password reset for targeted accounts.",
        "impossible_travel": "Verify user travel, require step-up authentication, and revoke active sessions if unconfirmed.",
        "lateral_movement": "Isolate the entity's device and investigate the sequence of accessed resources.",
        "device_spoofing": "Isolate the device, invalidate the device fingerprint, and re-enroll via trusted process.",
        "low_and_slow_exfiltration": "Investigate accessed resources, review recent transfer volumes, and monitor closely.",
        "insider_drift": "Monitor without blocking; review privilege/resource footprint growth with the entity's manager.",
        "normal": "No action required. Continue routine monitoring.",
    }
    action = actions.get(attack_type, "Investigate the flagged session manually.")
    if risk_lvl == "Critical" and attack_type != "normal":
        action = "URGENT - " + action + " Escalate to on-call security analyst immediately."
    return action
