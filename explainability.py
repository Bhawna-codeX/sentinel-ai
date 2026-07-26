"""
explainability.py
SentinelAI - rule-based explainability engine.

Generates human-readable alert explanations and ranks top contributing
factors. Uses transparent rule-based attribution (feature deviation vs.
baseline) rather than SHAP, to avoid extra runtime dependencies and keep
the hackathon MVP fast and stable, as required by the project constraints.
"""

FACTOR_LABELS = {
    "anomaly_score": "overall ML anomaly score",
    "failed_login_velocity": "elevated failed-login attempts",
    "geo_velocity": "abnormally high geographic velocity",
    "device_novelty": "unseen / mismatched device",
    "location_novelty": "login from an unfamiliar location",
    "resource_novelty": "access to an unusual resource",
    "privilege_deviation": "privilege level above the entity's norm",
    "off_hours": "activity outside typical hours",
    "bytes_deviation": "unusual data transfer volume",
    "classifier_confidence": "high model confidence in attack classification",
}


def top_contributing_factors(components, n=5):
    ranked = sorted(components.items(), key=lambda kv: kv[1], reverse=True)
    factors = [(FACTOR_LABELS.get(k, k), round(v, 2)) for k, v in ranked[:n] if v > 0.05]
    if not factors:
        factors = [(FACTOR_LABELS.get(ranked[0][0], ranked[0][0]), round(ranked[0][1], 2))]
    return factors


def build_explanation(row, attack_type, risk_score, risk_lvl, components, cold_start=False):
    factors = top_contributing_factors(components, n=5)
    factor_phrases = [f[0] for f in factors]

    cold_note = " This is a cold-start entity with limited history, so confidence is reduced." if cold_start else ""

    if attack_type == "normal" or risk_lvl == "Low":
        text = (
            f"Low-risk event (score {risk_score}/100). Behavior is broadly consistent with the "
            f"entity's baseline profile.{cold_note}"
        )
    else:
        phrase_str = ", ".join(factor_phrases[:-1])
        if len(factor_phrases) > 1:
            phrase_str += f", and {factor_phrases[-1]}"
        else:
            phrase_str = factor_phrases[0]
        attack_label = attack_type.replace("_", " ")
        text = (
            f"{risk_lvl} alert: classified as '{attack_label}' with a risk score of {risk_score}/100. "
            f"The session was flagged due to {phrase_str}.{cold_note}"
        )

    return {
        "narrative": text,
        "top_factors": factors,
    }
