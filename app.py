"""
app.py
SentinelAI - AI-Powered Behavioral Anomaly Detection for Cybersecurity
Streamlit SOC analyst dashboard.

Run:
    streamlit run app.py
"""

import os
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    BASE_DIR, DATA_DIR, MODEL_DIR, OUTPUT_DIR, RAW_CSV, TRAIN_CSV, TEST_CSV,
    BASELINE_JSON, FEATURE_COLUMNS, load_baselines, engineer_features, get_feature_matrix,
)
from risk_engine import compute_risk_score, risk_level, recommended_action, WEIGHTS
from explainability import build_explanation
from drift_detection import compute_drift_report, cold_start_entities, PROFILE_UPDATE_POLICY, TRUSTED_WINDOW_EVENTS

# --------------------------------------------------------------------------
# Page config & style
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="SentinelAI | Behavioral Anomaly Detection",
    page_icon="\U0001F6E1\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
.stApp { background-color: #0b1120; color: #d7e2f2; }
section[data-testid="stSidebar"] { background-color: #0e1626; border-right: 1px solid #1f2c42; }
h1, h2, h3, h4 { color: #e6edf7 !important; }
div[data-testid="stMetric"] {
    background: #101a2c; border: 1px solid #1f2c42; border-radius: 10px; padding: 12px 16px;
}
div[data-testid="stMetricValue"] { color: #3fa9f5 !important; }
.risk-critical { color: #ff4d4f; font-weight: 700; }
.risk-high { color: #ff9f43; font-weight: 700; }
.risk-medium { color: #f7d154; font-weight: 700; }
.risk-low { color: #3ddc84; font-weight: 700; }
.sentinel-card {
    background: #101a2c; border: 1px solid #1f2c42; border-radius: 12px;
    padding: 18px 20px; margin-bottom: 14px;
}
.sentinel-badge {
    display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.78rem;
    background:#1f2c42; color:#9fb3cc; margin-right:6px;
}
hr { border-color: #1f2c42; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

RISK_COLORS = {"Critical": "#ff4d4f", "High": "#ff9f43", "Medium": "#f7d154", "Low": "#3ddc84"}


# --------------------------------------------------------------------------
# Data / model loading (cached)
# --------------------------------------------------------------------------
def _files_exist(paths):
    return all(os.path.exists(p) for p in paths)


@st.cache_data(show_spinner="Loading synthetic access logs ...")
def load_raw_data():
    return pd.read_csv(RAW_CSV, parse_dates=["timestamp"])


@st.cache_resource(show_spinner="Loading trained models ...")
def load_models():
    iso = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    clf = joblib.load(os.path.join(MODEL_DIR, "attack_classifier.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    meta = joblib.load(os.path.join(MODEL_DIR, "best_classifier_meta.pkl"))
    return iso, clf, scaler, encoders, feature_cols, meta


@st.cache_data(show_spinner="Scoring events (Isolation Forest + attack classifier) ...")
def score_dataset(df):
    iso, clf, scaler, encoders, feature_cols, meta = load_models()
    feat_df = engineer_features(df)
    X = get_feature_matrix(feat_df, encoders)
    Xs = scaler.transform(X)

    iso_raw = -iso.score_samples(Xs)
    iso_norm = (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min() + 1e-9)

    pred_attack = clf.predict(X)
    proba = clf.predict_proba(X)
    classes = clf.classes_
    confidence = proba.max(axis=1)

    scores, levels, factors_list, narratives, actions = [], [], [], [], []
    for i in range(len(feat_df)):
        row = feat_df.iloc[i]
        cls_conf = confidence[i] if pred_attack[i] != "normal" else 0.0
        score, components = compute_risk_score(row, iso_norm[i], cls_conf)
        lvl = risk_level(score)
        cold = bool(row.get("cold_start_indicator", 0))
        expl = build_explanation(row, pred_attack[i], score, lvl, components, cold_start=cold)
        scores.append(score); levels.append(lvl)
        factors_list.append(json.dumps(expl["top_factors"]))
        narratives.append(expl["narrative"])
        actions.append(recommended_action(pred_attack[i], lvl))

    feat_df = feat_df.copy()
    feat_df["ml_anomaly_score"] = iso_norm
    feat_df["predicted_attack_type"] = pred_attack
    feat_df["model_confidence"] = confidence
    feat_df["risk_score"] = scores
    feat_df["risk_level"] = levels
    feat_df["top_factors"] = factors_list
    feat_df["explanation"] = narratives
    feat_df["recommended_action"] = actions
    return feat_df


def ensure_pipeline_ready():
    """Regenerate data / retrain models if artifacts are missing (robustness requirement)."""
    data_files = [RAW_CSV, TRAIN_CSV, TEST_CSV, BASELINE_JSON]
    model_files = [
        os.path.join(MODEL_DIR, f) for f in
        ["isolation_forest.pkl", "attack_classifier.pkl", "scaler.pkl", "encoders.pkl",
         "feature_columns.pkl", "best_classifier_meta.pkl"]
    ]
    if not _files_exist(data_files):
        with st.spinner("First run detected: generating synthetic dataset ..."):
            subprocess.run([sys.executable, os.path.join(BASE_DIR, "generate_data.py")], check=True)
        st.cache_data.clear()
    if not _files_exist(model_files):
        with st.spinner("First run detected: training models (this takes ~30-60s) ..."):
            subprocess.run([sys.executable, os.path.join(BASE_DIR, "train_models.py")], check=True)
        st.cache_resource.clear()


ensure_pipeline_ready()

# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.sidebar.markdown("## \U0001F6E1\ufe0f SentinelAI")
st.sidebar.caption("Behavioral Anomaly Detection for Cybersecurity")
st.sidebar.markdown("---")

PAGES = [
    "Executive Overview", "Live Alert Center", "Entity Behavioral Profile",
    "Threat Analytics", "Model Performance", "Drift & Cold Start",
    "Attack Simulator", "Architecture & About",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
if st.sidebar.button("\U0001F504 Regenerate synthetic data"):
    with st.spinner("Regenerating data ..."):
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "generate_data.py")], check=True)
    st.cache_data.clear()
    st.success("Data regenerated. Reloading ...")
    st.rerun()

if st.sidebar.button("\U0001F9E0 Retrain models"):
    with st.spinner("Retraining models ..."):
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "train_models.py")], check=True)
    st.cache_resource.clear()
    st.cache_data.clear()
    st.success("Models retrained. Reloading ...")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.warning("\u26A0\ufe0f All data is **synthetically generated** for this hackathon prototype. Metrics reflect synthetic-data performance only.")

# --------------------------------------------------------------------------
# Load data + score
# --------------------------------------------------------------------------
raw_df = load_raw_data()
scored_df = score_dataset(raw_df)

if "sim_events" not in st.session_state:
    st.session_state.sim_events = pd.DataFrame()

display_df = pd.concat([scored_df, st.session_state.sim_events], ignore_index=True) if len(st.session_state.sim_events) else scored_df

# ==========================================================================
# PAGE 1 - EXECUTIVE OVERVIEW
# ==========================================================================
if page == "Executive Overview":
    st.title("Executive Overview")
    st.caption("Synthetic prototype data - for demonstration purposes only.")

    total_events = len(display_df)
    total_anomalies = int((display_df["predicted_attack_type"] != "normal").sum())
    critical_alerts = int((display_df["risk_level"] == "Critical").sum())
    unique_entities = display_df["entity_id"].nunique()
    anomaly_rate = total_anomalies / max(total_events, 1) * 100
    mean_risk = display_df["risk_score"].mean()

    meta = joblib.load(os.path.join(MODEL_DIR, "best_classifier_meta.pkl"))
    eval_df = pd.read_csv(os.path.join(OUTPUT_DIR, "evaluation_metrics.csv"))
    iso_row = eval_df[(eval_df["model"] == "isolation_forest")].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Events", f"{total_events:,}")
    c2.metric("Detected Anomalies", f"{total_anomalies:,}")
    c3.metric("Critical Alerts", f"{critical_alerts:,}")
    c4.metric("Affected Entities", f"{unique_entities:,}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
    c6.metric("Mean Risk Score", f"{mean_risk:.1f}/100")
    c7.metric("Primary Model", "Isolation Forest")
    c8.metric("Top-1% Alert Budget Precision", f"{iso_row['top1pct_alert_budget_precision']*100:.1f}%")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Alert Timeline")
        ts_df = display_df.copy()
        ts_df["date"] = pd.to_datetime(ts_df["timestamp"]).dt.date
        timeline = ts_df[ts_df["predicted_attack_type"] != "normal"].groupby("date").size().reset_index(name="alerts")
        fig = px.line(timeline, x="date", y="alerts", markers=True)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", plot_bgcolor="#0b1120", height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Attack-Type Distribution")
        dist = display_df[display_df["predicted_attack_type"] != "normal"]["predicted_attack_type"].value_counts().reset_index()
        dist.columns = ["attack_type", "count"]
        fig = px.bar(dist, x="attack_type", y="count", color="attack_type")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", plot_bgcolor="#0b1120", height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Risk-Level Distribution")
        risk_dist = display_df["risk_level"].value_counts().reindex(["Low", "Medium", "High", "Critical"]).fillna(0).reset_index()
        risk_dist.columns = ["risk_level", "count"]
        fig = px.pie(risk_dist, names="risk_level", values="count",
                     color="risk_level", color_discrete_map=RISK_COLORS, hole=0.45)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.subheader("Normal vs. Anomalous Events")
        nv = display_df["predicted_attack_type"].apply(lambda x: "Normal" if x == "normal" else "Anomalous").value_counts().reset_index()
        nv.columns = ["status", "count"]
        fig = px.bar(nv, x="status", y="count", color="status",
                     color_discrete_map={"Normal": "#3fa9f5", "Anomalous": "#e05263"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", plot_bgcolor="#0b1120", height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col_e, col_f = st.columns(2)
    with col_e:
        st.subheader("Top Suspicious Entities")
        top_ent = (display_df[display_df["predicted_attack_type"] != "normal"]
                   .groupby("entity_id")["risk_score"].agg(["mean", "count"])
                   .sort_values("count", ascending=False).head(10).reset_index())
        top_ent.columns = ["entity_id", "avg_risk_score", "alert_count"]
        fig = px.bar(top_ent, x="entity_id", y="alert_count", color="avg_risk_score",
                     color_continuous_scale="Reds")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", plot_bgcolor="#0b1120", height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_f:
        st.subheader("Top Suspicious Source IPs")
        top_ip = (display_df[display_df["predicted_attack_type"] != "normal"]
                  ["source_ip"].value_counts().head(10).reset_index())
        top_ip.columns = ["source_ip", "alert_count"]
        fig = px.bar(top_ip, x="source_ip", y="alert_count", color="alert_count", color_continuous_scale="Oranges")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", plot_bgcolor="#0b1120", height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Hourly Anomaly Heatmap")
    heat_df = display_df[display_df["predicted_attack_type"] != "normal"].copy()
    heat_df["day_of_week"] = pd.to_datetime(heat_df["timestamp"]).dt.day_name()
    pivot = heat_df.pivot_table(index="day_of_week", columns="hour", values="event_id", aggfunc="count", fill_value=0)
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex(day_order).fillna(0)
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Turbo", labels=dict(x="Hour", y="Day", color="Alerts"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", height=380)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================================
# PAGE 2 - LIVE ALERT CENTER
# ==========================================================================
elif page == "Live Alert Center":
    st.title("Live Alert Center")
    st.caption("Ranked, filterable list of flagged events with full explainability.")

    with st.expander("Filters", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        min_date = pd.to_datetime(display_df["timestamp"]).min().date()
        max_date = pd.to_datetime(display_df["timestamp"]).max().date()
        date_range = f1.date_input("Date range", (min_date, max_date), min_value=min_date, max_value=max_date)
        risk_filter = f2.multiselect("Risk level", ["Low", "Medium", "High", "Critical"], default=["High", "Critical"])
        entity_type_filter = f3.multiselect("Entity type", sorted(display_df["entity_type"].unique()), default=list(sorted(display_df["entity_type"].unique())))
        attack_filter = f4.multiselect("Attack type", sorted(display_df["predicted_attack_type"].unique()),
                                        default=[a for a in display_df["predicted_attack_type"].unique() if a != "normal"])
        f5, f6 = st.columns(2)
        country_filter = f5.multiselect("Country", sorted(display_df["country"].unique()), default=[])
        min_risk = f6.slider("Minimum risk score", 0, 100, 0)

    filtered = display_df.copy()
    filtered["timestamp"] = pd.to_datetime(filtered["timestamp"])
    if isinstance(date_range, tuple) and len(date_range) == 2:
        filtered = filtered[(filtered["timestamp"].dt.date >= date_range[0]) & (filtered["timestamp"].dt.date <= date_range[1])]
    if risk_filter:
        filtered = filtered[filtered["risk_level"].isin(risk_filter)]
    if entity_type_filter:
        filtered = filtered[filtered["entity_type"].isin(entity_type_filter)]
    if attack_filter:
        filtered = filtered[filtered["predicted_attack_type"].isin(attack_filter)]
    if country_filter:
        filtered = filtered[filtered["country"].isin(country_filter)]
    filtered = filtered[filtered["risk_score"] >= min_risk]
    filtered = filtered.sort_values("risk_score", ascending=False)

    st.markdown(f"**{len(filtered):,} alerts match current filters.**")

    show_cols = ["timestamp", "entity_id", "entity_type", "source_ip", "country", "resource_accessed",
                 "predicted_attack_type", "model_confidence", "risk_score", "risk_level"]
    st.dataframe(
        filtered[show_cols].head(500).style.format({"model_confidence": "{:.2f}"}),
        use_container_width=True, height=380,
    )

    st.download_button(
        "\U0001F4E5 Download filtered alerts as CSV",
        filtered[show_cols].to_csv(index=False).encode("utf-8"),
        "sentinelai_filtered_alerts.csv", "text/csv",
    )

    st.markdown("---")
    st.subheader("Alert Detail")
    if len(filtered) == 0:
        st.info("No alerts match the current filters.")
    else:
        options = filtered["event_id"].astype(str) + " | " + filtered["entity_id"] + " | " + filtered["predicted_attack_type"]
        selected = st.selectbox("Select an alert to inspect", options.tolist())
        sel_event_id = int(selected.split(" | ")[0])
        row = filtered[filtered["event_id"] == sel_event_id].iloc[0]

        risk_css = f"risk-{row['risk_level'].lower()}"
        st.markdown(f"""
        <div class="sentinel-card">
        <span class="sentinel-badge">Entity: {row['entity_id']}</span>
        <span class="sentinel-badge">Type: {row['entity_type']}</span>
        <span class="sentinel-badge">Attack: {row['predicted_attack_type']}</span>
        <h3 class="{risk_css}">Risk Score: {row['risk_score']}/100 ({row['risk_level']})</h3>
        <p>{row['explanation']}</p>
        </div>
        """, unsafe_allow_html=True)

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Event Details**")
            detail_fields = ["timestamp", "source_ip", "country", "city", "resource_accessed", "auth_method",
                              "login_status", "device_id", "device_os", "protocol", "bytes_transferred",
                              "session_duration_minutes", "privilege_level", "failed_attempts_last_10_min",
                              "geo_velocity_kmph"]
            st.table(row[detail_fields].astype(str))

        with d2:
            st.markdown("**Top Contributing Factors**")
            factors = json.loads(row["top_factors"])
            for name, val in factors:
                st.progress(min(val, 1.0), text=f"{name} ({val:.2f})")

            st.markdown("**Recommended Analyst Action**")
            st.info(row["recommended_action"])

            baselines = load_baselines()
            b = baselines.get(row["entity_id"])
            if b:
                st.markdown("**Entity Baseline Comparison**")
                comp = pd.DataFrame({
                    "Feature": ["Typical login hour", "Typical city", "Typical device", "Typical auth", "Typical privilege"],
                    "Baseline": [b["login_hour_center"], b["city"], b["device_id"], b["auth_method"], b["privilege_level"]],
                    "This event": [row["hour"], row["city"], row["device_id"], row["auth_method"], row["privilege_level"]],
                })
                st.table(comp)

        st.markdown("**Entity Activity Timeline (recent events)**")
        ent_timeline = display_df[display_df["entity_id"] == row["entity_id"]].sort_values("timestamp").tail(30)
        fig = px.scatter(ent_timeline, x="timestamp", y="risk_score", color="risk_level",
                          color_discrete_map=RISK_COLORS, hover_data=["predicted_attack_type"])
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", plot_bgcolor="#0b1120", height=320)
        st.plotly_chart(fig, use_container_width=True)

# ==========================================================================
# PAGE 3 - ENTITY BEHAVIORAL PROFILE
# ==========================================================================
elif page == "Entity Behavioral Profile":
    st.title("Entity Behavioral Profile")
    st.caption("Inspect any user, service account, or device's learned baseline behavior.")

    baselines = load_baselines()
    entity_ids = sorted(display_df["entity_id"].unique())
    selected_entity = st.selectbox("Select entity", entity_ids)

    b = baselines.get(selected_entity, {})
    ent_events = display_df[display_df["entity_id"] == selected_entity].sort_values("timestamp")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Events", len(ent_events))
    c2.metric("Mean Risk Score", f"{ent_events['risk_score'].mean():.1f}" if len(ent_events) else "N/A")
    c3.metric("Cold-Start Entity", "Yes" if len(ent_events) and ent_events['cold_start_indicator'].iloc[-1] else "No")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Baseline Profile")
        if b:
            st.markdown(f"""
            <div class="sentinel-card">
            <b>Typical login hour:</b> {b['login_hour_center']}:00 (&plusmn;{b['login_hour_spread']:.1f}h)<br>
            <b>Known location:</b> {b['city']}, {b['country']}<br>
            <b>Known device:</b> {b['device_id']} ({b['device_os']})<br>
            <b>Common auth method:</b> {b['auth_method']}<br>
            <b>Common resources:</b> {', '.join(b['typical_resources'])}<br>
            <b>Session duration baseline:</b> {b['session_duration_mean']:.1f} min (&plusmn;{b['session_duration_std']:.1f})<br>
            <b>Bytes-transfer baseline:</b> {b['bytes_mean']:,.0f} bytes (&plusmn;{b['bytes_std']:,.0f})<br>
            <b>Typical privilege level:</b> {b['privilege_level']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No baseline profile found for this entity (cold-start).")

    with col_b:
        st.subheader("Current Deviation from Baseline")
        if len(ent_events):
            dev_cols = ["login_hour_deviation", "session_duration_deviation", "bytes_transfer_deviation",
                        "device_novelty_score", "location_novelty_score", "resource_novelty_score", "privilege_deviation"]
            latest = ent_events.iloc[-1][dev_cols]
            fig = go.Figure(data=go.Scatterpolar(r=latest.values, theta=dev_cols, fill="toself"))
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", height=350,
                               polar=dict(bgcolor="#101a2c"))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Historical Risk Trend")
    if len(ent_events):
        fig = px.line(ent_events, x="timestamp", y="risk_score", markers=True)
        fig.add_hline(y=60, line_dash="dash", line_color="orange")
        fig.add_hline(y=80, line_dash="dash", line_color="red")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", plot_bgcolor="#0b1120", height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Events")
    show_cols = ["timestamp", "resource_accessed", "device_id", "city", "predicted_attack_type", "risk_score", "risk_level"]
    st.dataframe(ent_events[show_cols].tail(50).sort_values("timestamp", ascending=False), use_container_width=True, height=300)

# ==========================================================================
# PAGE 4 - THREAT ANALYTICS
# ==========================================================================
elif page == "Threat Analytics":
    st.title("Threat Analytics")
    st.caption("Deep-dive analysis for each supported attack category.")

    anomalies = display_df[display_df["predicted_attack_type"] != "normal"]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Brute Force & Credential Stuffing", "Impossible Travel", "Lateral Movement",
        "Device Spoofing", "Exfiltration & Insider Drift",
    ])

    with tab1:
        bf = anomalies[anomalies["predicted_attack_type"] == "brute_force"]
        cs = anomalies[anomalies["predicted_attack_type"] == "credential_stuffing"]
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Brute-Force Trend (failed attempts)")
            fig = px.scatter(bf, x="timestamp", y="failed_attempts_last_10_min", color="risk_score",
                              color_continuous_scale="Reds", hover_data=["entity_id"])
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", height=340)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Credential-Stuffing Source-IP Analysis")
            ip_counts = cs["source_ip"].value_counts().reset_index()
            ip_counts.columns = ["source_ip", "attempts"]
            fig = px.bar(ip_counts.head(15), x="source_ip", y="attempts", color="attempts", color_continuous_scale="Oranges")
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", height=340)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        it = anomalies[anomalies["predicted_attack_type"] == "impossible_travel"]
        st.subheader(f"Impossible-Travel Cases ({len(it)} detected)")
        if len(it):
            fig = px.scatter_geo(it, lat="latitude", lon="longitude", color="risk_score",
                                  hover_name="entity_id", color_continuous_scale="Reds", projection="natural earth")
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", height=420, geo=dict(bgcolor="#0b1120"))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(it[["timestamp", "entity_id", "city", "country", "geo_velocity_kmph", "risk_score"]],
                         use_container_width=True, height=250)
        else:
            st.info("No impossible-travel cases in the current dataset.")

    with tab3:
        lm = anomalies[anomalies["predicted_attack_type"] == "lateral_movement"]
        st.subheader("Lateral-Movement Resource-Access Pattern")
        if len(lm):
            trans = lm.groupby(["entity_id", "resource_category"]).size().reset_index(name="accesses")
            fig = px.sunburst(trans, path=["entity_id", "resource_category"], values="accesses",
                               color="accesses", color_continuous_scale="Reds")
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No lateral-movement cases in the current dataset.")

    with tab4:
        ds = anomalies[anomalies["predicted_attack_type"] == "device_spoofing"]
        st.subheader(f"Device-Fingerprint Mismatches ({len(ds)} detected)")
        st.dataframe(ds[["timestamp", "entity_id", "device_id", "device_os", "protocol", "risk_score"]],
                     use_container_width=True, height=320)

    with tab5:
        ex = anomalies[anomalies["predicted_attack_type"] == "low_and_slow_exfiltration"]
        idr = anomalies[anomalies["predicted_attack_type"] == "insider_drift"]
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Low-and-Slow Exfiltration Trend")
            if len(ex):
                fig = px.line(ex.sort_values("timestamp"), x="timestamp", y="bytes_transferred", markers=True)
                fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", height=340)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No exfiltration cases detected.")
        with col2:
            st.subheader("Insider-Drift Cases (privilege/resource growth)")
            if len(idr):
                fig = px.scatter(idr, x="entity_history_length", y="privilege_deviation", color="risk_score",
                                  color_continuous_scale="Reds", hover_data=["entity_id"])
                fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", height=340)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No insider-drift cases detected.")

    st.markdown("---")
    st.subheader("Attack Distribution by Entity Type")
    dist = anomalies.groupby(["entity_type", "predicted_attack_type"]).size().reset_index(name="count")
    fig = px.bar(dist, x="entity_type", y="count", color="predicted_attack_type", barmode="stack")
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", plot_bgcolor="#0b1120", height=380)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================================
# PAGE 5 - MODEL PERFORMANCE
# ==========================================================================
elif page == "Model Performance":
    st.title("Model Performance")
    st.caption("All metrics are computed on held-out synthetic test data (chronological split). "
               "They demonstrate methodology, not production-grade guarantees.")

    eval_df = pd.read_csv(os.path.join(OUTPUT_DIR, "evaluation_metrics.csv"))
    st.subheader("Model Comparison")
    st.dataframe(eval_df, use_container_width=True)

    st.info(
        "**Why accuracy alone is misleading:** with ~97% of events being normal, a model that "
        "predicts 'normal' for everything would score ~97% accuracy while catching zero attacks. "
        "This is why SentinelAI reports precision, recall, F1, PR-AUC, and top-alert-budget "
        "precision alongside accuracy."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix (Attack Classifier)")
        cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
    with col2:
        st.subheader("Feature Importance")
        fi_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
        if os.path.exists(fi_path):
            st.image(fi_path, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Anomaly Score Distribution")
        as_path = os.path.join(OUTPUT_DIR, "anomaly_score_distribution.png")
        if os.path.exists(as_path):
            st.image(as_path, use_container_width=True)
    with col4:
        st.subheader("Per-Class Report (Best Classifier)")
        pc_path = os.path.join(OUTPUT_DIR, "per_class_report.csv")
        if os.path.exists(pc_path):
            st.dataframe(pd.read_csv(pc_path, index_col=0), use_container_width=True, height=340)

    st.subheader("Top-1% Alert-Budget Results")
    st.caption("Precision achieved if analysts only investigate the top 1% highest-scored events - "
               "a realistic SOC operating constraint.")
    budget_df = eval_df[eval_df["stage"] == "anomaly_detection"][
        ["model", "top1pct_alert_budget_precision", "alert_budget_k"]
    ]
    st.table(budget_df)

    st.download_button(
        "\U0001F4E5 Download evaluation metrics (CSV)",
        eval_df.to_csv(index=False).encode("utf-8"),
        "evaluation_metrics.csv", "text/csv",
    )

# ==========================================================================
# PAGE 6 - DRIFT AND COLD START
# ==========================================================================
elif page == "Drift & Cold Start":
    st.title("Concept Drift & Cold-Start Monitoring")

    st.markdown("""
    <div class="sentinel-card">
    <b>Concept drift method:</b> Population Stability Index (PSI) comparing an older "baseline"
    window of events against a more recent window, per feature. PSI &lt; 0.1 = stable,
    0.1-0.25 = moderate drift, &gt; 0.25 = significant drift and worth investigating.
    </div>
    """, unsafe_allow_html=True)

    drift_df = compute_drift_report(display_df)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Drift Score by Feature (PSI)")
        fig = px.bar(drift_df, x="feature", y="psi", color="status",
                     color_discrete_map={"Stable": "#3ddc84", "Moderate Drift": "#f7d154", "Significant Drift": "#ff4d4f"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", plot_bgcolor="#0b1120", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Drift Table")
        st.dataframe(drift_df, use_container_width=True, height=380)

    st.subheader("Historical vs. Recent Distribution (example feature)")
    example_feature = st.selectbox("Feature", drift_df["feature"].tolist())
    df_sorted = display_df.sort_values("timestamp")
    split_idx = int(len(df_sorted) * 0.5)
    base_vals = df_sorted.iloc[:split_idx][example_feature]
    recent_vals = df_sorted.iloc[split_idx:][example_feature]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=base_vals, name="Historical baseline", opacity=0.6, marker_color="#3fa9f5"))
    fig.add_trace(go.Histogram(x=recent_vals, name="Recent window", opacity=0.6, marker_color="#e05263"))
    fig.update_layout(barmode="overlay", template="plotly_dark", paper_bgcolor="#0b1120", height=340)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Cold-Start Entities")
    cold_df = cold_start_entities(display_df, threshold=5)
    n_cold = int(cold_df["cold_start"].sum())
    st.metric("Entities With Insufficient History", n_cold)
    st.dataframe(cold_df, use_container_width=True, height=280)

    st.markdown("---")
    st.subheader("Profile-Update Policy")
    st.info(PROFILE_UPDATE_POLICY)
    st.caption(f"Trusted observation window: {TRUSTED_WINDOW_EVENTS} consecutive low-risk events before a baseline is updated.")

# ==========================================================================
# PAGE 7 - ATTACK SIMULATOR
# ==========================================================================
elif page == "Attack Simulator":
    st.title("Attack Simulator")
    st.caption("Generate a synthetic event with a chosen attack pattern, score it live, and see the full explanation.")

    from generate_data import (
        attack_event, normal_event, build_baselines, make_entity_pool, ATTACK_TYPES,
    )

    baselines = load_baselines()
    entity_ids = sorted(display_df["entity_id"].unique())

    c1, c2, c3 = st.columns(3)
    sim_attack = c1.selectbox("Attack type", ATTACK_TYPES)
    sim_entity = c2.selectbox("Entity ID", entity_ids)
    sim_severity = c3.select_slider("Severity", ["Mild", "Typical", "Severe"], value="Typical")

    if st.button("\u26A1 Simulate Attack", type="primary"):
        b = baselines.get(sim_entity)
        if b is None:
            st.error("No baseline found for this entity.")
        else:
            severity_mult = {"Mild": 0.6, "Typical": 1.0, "Severe": 1.6}[sim_severity]
            next_id = int(display_df["event_id"].max()) + len(st.session_state.sim_events) + 1
            row = attack_event(sim_entity, b, next_id, sim_attack)

            # apply severity scaling to a few key signals
            row["failed_attempts_last_10_min"] = int(row["failed_attempts_last_10_min"] * severity_mult)
            row["geo_velocity_kmph"] = round(row["geo_velocity_kmph"] * severity_mult, 2)
            row["bytes_transferred"] = round(row["bytes_transferred"] * severity_mult, 2)

            sim_df = pd.DataFrame([row])
            sim_df["hour"] = pd.to_datetime(sim_df["timestamp"]).dt.hour
            sim_df["day_of_week"] = pd.to_datetime(sim_df["timestamp"]).dt.dayofweek
            sim_df["is_weekend"] = (sim_df["day_of_week"] >= 5).astype(int)
            sim_df["unique_resources_last_24h"] = display_df[display_df["entity_id"] == sim_entity]["unique_resources_last_24h"].max() if len(display_df[display_df["entity_id"] == sim_entity]) else 1
            sim_df["time_since_last_event_minutes"] = 5.0
            sim_df["entity_history_length"] = len(display_df[display_df["entity_id"] == sim_entity]) + 1

            iso, clf, scaler, encoders, feature_cols, meta = load_models()
            hist_count = {sim_entity: len(display_df[display_df["entity_id"] == sim_entity])}
            feat_df = engineer_features(sim_df, baselines=baselines, entity_history=hist_count)
            X = get_feature_matrix(feat_df, encoders)
            Xs = scaler.transform(X)

            iso_raw = -iso.score_samples(Xs)
            # normalize against the existing dataset's range for comparability
            ref_scores = display_df["ml_anomaly_score"] if "ml_anomaly_score" in display_df else pd.Series([0, 1])
            iso_norm = float(np.clip((iso_raw[0] - iso_raw.min()) / (max(iso_raw.max() - iso_raw.min(), 1e-9)), 0, 1))
            iso_norm = float(np.clip(0.5 + iso_norm * 0.5, 0, 1))  # simulated single events skew high; keep interpretable

            pred = clf.predict(X)[0]
            proba = clf.predict_proba(X)[0]
            conf = float(proba.max())

            frow = feat_df.iloc[0]
            score, components = compute_risk_score(frow, iso_norm, conf if pred != "normal" else 0.0)
            lvl = risk_level(score)
            cold = bool(frow.get("cold_start_indicator", 0))
            expl = build_explanation(frow, pred, score, lvl, components, cold_start=cold)
            action = recommended_action(pred, lvl)

            feat_df["ml_anomaly_score"] = iso_norm
            feat_df["predicted_attack_type"] = pred
            feat_df["model_confidence"] = conf
            feat_df["risk_score"] = score
            feat_df["risk_level"] = lvl
            feat_df["top_factors"] = json.dumps(expl["top_factors"])
            feat_df["explanation"] = expl["narrative"]
            feat_df["recommended_action"] = action

            st.session_state.sim_events = pd.concat([st.session_state.sim_events, feat_df], ignore_index=True)

            risk_css = f"risk-{lvl.lower()}"
            st.markdown(f"""
            <div class="sentinel-card">
            <h3 class="{risk_css}">Risk Score: {score}/100 ({lvl})</h3>
            <p><b>Predicted attack type:</b> {pred} &nbsp;|&nbsp; <b>Model confidence:</b> {conf:.2f}</p>
            <p>{expl['narrative']}</p>
            <p><b>Recommended action:</b> {action}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Top contributing factors**")
            for name, val in expl["top_factors"]:
                st.progress(min(val, 1.0), text=f"{name} ({val:.2f})")

            st.success("Event added to this session's Live Alert Center feed.")

    if len(st.session_state.sim_events):
        st.markdown("---")
        st.subheader("Simulated Events (this session)")
        st.dataframe(
            st.session_state.sim_events[["timestamp", "entity_id", "predicted_attack_type", "risk_score", "risk_level"]],
            use_container_width=True,
        )
        if st.button("Clear simulated events"):
            st.session_state.sim_events = pd.DataFrame()
            st.rerun()

# ==========================================================================
# PAGE 8 - ARCHITECTURE AND ABOUT
# ==========================================================================
elif page == "Architecture & About":
    st.title("Architecture & About")

    st.subheader("Problem Statement")
    st.write(
        "Design an AI/ML system that learns the normal access and connection behavior of users, "
        "service accounts, and devices; detects suspicious behavior in near real time; classifies "
        "the probable anomaly type; generates an explainable risk score; and provides an "
        "analyst-facing dashboard - while handling class imbalance, concept drift, cold-start "
        "entities, and real-time streaming feasibility."
    )

    st.subheader("Architecture")
    stages = [
        "Synthetic Access Log Generator", "Data Validation & Preprocessing", "Feature Engineering",
        "Per-Entity Behavioral Baselines", "Anomaly Detection Model (Isolation Forest)",
        "Attack-Type Classifier (Random Forest)", "Risk & Explainability Engine",
        "Concept Drift Monitor", "Streamlit SOC Dashboard",
    ]
    fig = go.Figure()
    for i, stage in enumerate(stages):
        fig.add_shape(type="rect", x0=0, x1=6, y0=-i, y1=-i - 0.8,
                      line=dict(color="#3fa9f5"), fillcolor="#101a2c")
        fig.add_annotation(x=3, y=-i - 0.4, text=stage, showarrow=False, font=dict(color="#e6edf7", size=13))
        if i < len(stages) - 1:
            fig.add_annotation(x=3, y=-i - 0.9, ax=3, ay=-i - 0.8, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowcolor="#3fa9f5")
    fig.update_xaxes(visible=False, range=[-1, 7])
    fig.update_yaxes(visible=False, range=[-len(stages) - 0.5, 1])
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1120", plot_bgcolor="#0b1120",
                      height=620, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Additional components: model artifacts (models/), historical entity-profile storage "
               "(data/entity_baselines.json), an analyst feedback loop (planned future extension), "
               "and a real-time deployment extension path (batch -> micro-batch -> streaming).")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Technology Stack")
        st.markdown("""
        - **Language:** Python 3
        - **Dashboard:** Streamlit
        - **Data:** Pandas, NumPy, Faker (synthetic generation)
        - **ML:** Scikit-learn (Isolation Forest, One-Class SVM, LOF, Random Forest, HistGradientBoosting, Logistic Regression)
        - **Visualization:** Plotly, Matplotlib
        - **Persistence:** Joblib
        - **Presentation:** python-pptx
        """)
    with col2:
        st.subheader("Limitations")
        st.markdown("""
        - Trained entirely on **synthetic** data; real traffic will differ in distribution and volume.
        - Risk-score weights are heuristic and interpretable, not statistically optimized.
        - No true real-time streaming (Kafka etc.) - this MVP simulates near-real-time via on-demand scoring.
        - Rule-based explainability instead of SHAP, for stability and speed in a hackathon setting.
        - Single-node prototype; not load-tested at enterprise scale.
        """)

    st.subheader("Future Scope")
    st.markdown("""
    - Integrate a real streaming pipeline (Kafka/Flink) for true real-time scoring.
    - Add SHAP-based explainability once dependency/runtime constraints are no longer a concern.
    - Incorporate analyst feedback loops to retrain models on confirmed true/false positives.
    - Expand entity graph analysis for lateral-movement detection using graph neural networks.
    - Validate against real (anonymized) SOC datasets before any production consideration.
    """)

    st.markdown("---")
    st.caption("SentinelAI - Hackathon Prototype | All data shown is synthetically generated | Built with Streamlit")