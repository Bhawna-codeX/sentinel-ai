# SentinelAI - Project Report

**AI-Powered Behavioral Anomaly Detection for Cybersecurity**
**Prototype status: hackathon MVP, built entirely on synthetically generated data.**

---

## 1. Title Page

- **Project Title:** SentinelAI
- **Theme:** Cybersecurity
- **Category:** Software
- **Student:** Bhawna Chaurasia
- **Student ID:** 23BAI10924
- **Tagline:** "Learn normal. Detect deviations. Explain every alert."

---

## 2. Executive Summary

SentinelAI is a hackathon-scale prototype that demonstrates how a hybrid unsupervised/supervised
machine-learning pipeline can learn the normal behavior of users, service accounts, and devices,
and flag deviations that resemble common attacker techniques. The system generates its own
realistic synthetic access-log dataset, engineers behavioral features per entity, trains an
Isolation Forest anomaly detector and a Random Forest attack-type classifier, computes a
transparent 0-100 risk score for every event, and presents the results through an 8-page
Streamlit dashboard aimed at a SOC (Security Operations Center) analyst audience. The system also
demonstrates lightweight concept-drift monitoring and cold-start entity handling. All reported
metrics come directly from model predictions on held-out synthetic test data and are clearly
labeled as prototype-stage results.

## 3. Background

Modern organizations generate enormous volumes of access and authentication events across users,
automated service accounts, and edge devices. Traditional signature-based detection struggles to
catch novel or slow-moving attacker behavior, and SOC analysts are frequently overwhelmed by
low-quality alerts with no explanation of *why* something was flagged. Behavioral anomaly
detection - learning what is "normal" for a specific entity and flagging deviations - is a
widely used approach in User and Entity Behavior Analytics (UEBA), but building an MVP that
handles imbalance, drift, and explainability together is non-trivial.

## 4. Problem Statement

Design an AI/ML system that learns the normal access and connection behavior of users, service
accounts, and devices; detects suspicious behavior in near real time; classifies the probable
anomaly type; generates an explainable risk score; and provides an analyst-facing dashboard. The
system must specifically address: sequential/behavioral access data, extreme class imbalance,
concept drift, explainability, cold-start entities, and real-time streaming feasibility.

## 5. Objectives

1. Generate a realistic, reproducible synthetic access-log dataset with per-entity baselines.
2. Engineer behavioral features that capture deviation from each entity's normal profile.
3. Train and compare unsupervised anomaly-detection models under extreme class imbalance.
4. Train and compare supervised attack-type classifiers across 7 attack categories.
5. Produce a deterministic, explainable 0-100 risk score for every event.
6. Demonstrate concept-drift detection and cold-start entity handling.
7. Deliver an analyst-facing dashboard covering executive, investigative, and technical views.

## 6. Proposed Solution

A two-stage hybrid pipeline: Stage A uses Isolation Forest (primary) to flag statistically
unusual events without needing labels; Stage B uses a Random Forest classifier (primary) to
assign a probable attack category to flagged events. A rule-based risk-scoring engine combines
the anomaly score with concrete behavioral signals (failed logins, geo-velocity, device/location
novelty, privilege deviation, etc.) into a single, auditable 0-100 score. A rule-based
explainability layer converts the top contributing signals into a plain-language narrative for
each alert.

## 7. Innovation and Uniqueness

- **Per-entity behavioral baselines** rather than a single global model, so the system judges each
  user, service account, or device against its own history.
- **Deterministic, auditable risk scoring** instead of an opaque model output - every score can be
  traced back to specific weighted signals.
- **PSI-based concept-drift monitor** with an explicit trusted-observation-window policy so that
  baselines are never silently updated from a session that was flagged as high-risk.
- **Explicit cold-start handling** with confidence down-weighting rather than pretending new
  entities behave like established ones.

## 8. Synthetic Dataset Design

The dataset is generated with a fixed random seed for reproducibility and models 150 users, 20
service accounts, and 30 devices over roughly a four-month window, totaling 22,000 events. Each
entity is assigned its own baseline: typical login hours, home country/city, known device and
fingerprint, preferred authentication method, common resource set, typical session duration and
data-transfer volume, and typical privilege level. Statistical noise (drawn from normal
distributions) is layered onto every normal event so that behavior is realistic rather than
perfectly repetitive. Approximately 3% of events are anomalous, consistent with the requested
2-4% range for extreme class imbalance.

## 9. Attack Taxonomy

Seven attack categories are injected, each with a distinct behavioral signature:

| Attack Type | Core Signature |
|---|---|
| Brute Force | Rapid, repeated failed logins in a short window |
| Credential Stuffing | High failure rate across many entities from few source IPs |
| Impossible Travel | Two logins from geographically distant locations in an implausibly short time |
| Lateral Movement | Broader-than-normal resource access with elevated command counts / privilege |
| Device Spoofing | Known device ID paired with a mismatched fingerprint, OS, or protocol |
| Low-and-Slow Exfiltration | Off-hours, elevated but modest data transfer to sensitive resources |
| Insider Drift | Gradual privilege/resource-footprint expansion; an intentionally ambiguous edge case |

## 10. Feature Engineering

Engineered behavioral features include: login-hour deviation, session-duration deviation,
resource/device/location/authentication/protocol novelty scores, failed-login velocity, unique
resources accessed, bytes-transfer deviation, privilege deviation, geo-velocity, time since the
previous event, off-hours and weekend indicators, entity history length, and a cold-start
indicator. Categorical fields are encoded with unseen-category-safe frequency encoders; numeric
features are scaled with a `StandardScaler`. The `anomaly_label` and `attack_type` columns are
used only as training targets and are never included among the input features, preventing label
leakage. Training and test data are split chronologically (75%/25% by timestamp) rather than
randomly, approximating a realistic "train on the past, predict the future" evaluation.

## 11. Behavioral Profiling

Per-entity baselines are computed empirically from each entity's own historical events (mean/
standard deviation of login hour, session duration, bytes transferred; most common device,
location, authentication method, protocol, resource category, and privilege level). At inference/
simulation time, these baselines can be loaded from `data/entity_baselines.json` so that a single
new event can be scored without needing to reprocess the entire historical dataset.

## 12. Anomaly-Detection Methodology

Three unsupervised models were trained and compared on the same feature set:

| Model | Precision | Recall | F1 | PR-AUC | FPR | Top-1% Alert-Budget Precision |
|---|---|---|---|---|---|---|
| Isolation Forest (primary) | 0.325 | 0.526 | 0.402 | 0.387 | 0.036 | 0.600 |
| One-Class SVM | 0.290 | 0.886 | 0.437 | 0.706 | 0.071 | 0.946 |
| Local Outlier Factor | 0.346 | 0.680 | 0.459 | 0.657 | 0.042 | 1.000 |

Isolation Forest was selected as the primary production model per the project constraints because
it is fast to train and score (fit-time latency of roughly 0.02 ms/event in this prototype),
scales well to more data, and does not require the full training set to be held in memory at
inference time the way instance-based methods like LOF do. One-Class SVM and Local Outlier Factor
are retained as comparison baselines and, notably, show stronger precision within a realistic
top-1%-alert-budget - a useful finding documented here rather than hidden, since it illustrates a
genuine speed-vs-precision trade-off SOC teams face in practice.

## 13. Attack-Classification Methodology

Three supervised classifiers were trained with class weighting across all 8 classes (`normal` +
7 attack types):

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| Random Forest (selected) | 0.996 | 0.948 | 0.890 | 0.915 | 0.996 |
| HistGradientBoosting | 0.986 | 0.809 | 0.842 | 0.797 | 0.988 |
| Logistic Regression | 0.772 | 0.515 | 0.859 | 0.545 | 0.857 |

Random Forest was automatically selected as the best model by macro F1 (the metric that treats
every attack class equally regardless of its frequency, which matters far more than raw accuracy
given the dataset's imbalance). Full per-class precision/recall/F1 and the confusion matrix are
available in `outputs/per_class_report.csv` and `outputs/confusion_matrix.png`.

## 14. Risk-Scoring Methodology

The 0-100 risk score is a deterministic, weighted combination of ten normalized signals: the ML
anomaly score, failed-login velocity, geo-velocity, device/location/resource novelty, privilege
deviation, off-hours activity, bytes-transfer deviation, and attack-classifier confidence.
Weights were chosen heuristically for interpretability (see `risk_engine.py`) rather than
statistically fit, and are documented in full so any analyst can audit exactly why a given score
was produced. Score bands: 0-29 Low, 30-59 Medium, 60-79 High, 80-100 Critical.

## 15. Explainability

Rather than SHAP (which adds dependency and runtime overhead not well suited to a fast-moving
hackathon build), SentinelAI uses rule-based attribution: for every alert, the risk-engine's
component scores are ranked and the top 3-5 highest-scoring factors are converted into a
plain-language sentence (e.g., "flagged due to elevated failed-login attempts, access to an
unusual resource, and activity outside typical hours"). Each alert also surfaces the entity's
baseline values alongside the actual event values and a recommended analyst action appropriate to
the predicted attack type.

## 16. Concept Drift

Concept drift is measured using the Population Stability Index (PSI) between an older ("baseline")
window and a more recent window of the same features, computed on quantile-based buckets of the
baseline distribution. PSI < 0.1 is treated as stable, 0.1-0.25 as moderate drift, and > 0.25 as
significant drift. The dashboard's "Drift & Cold Start" page displays PSI per feature along with
overlaid historical-vs-recent distribution histograms. Behavioral baselines are only updated after
a trusted observation window of 30 consecutive low-risk events for a given entity, and sessions
flagged Critical or High risk are explicitly excluded from ever being used to update a baseline -
preventing an attacker's behavior from being silently "learned" as normal.

## 17. Cold-Start Handling

Entities with 5 or fewer historical events are flagged as cold-start. For these entities, the
system falls back to conservative default/global assumptions and reduces confidence in any
resulting alert rather than asserting high certainty from minimal history. Cold-start status is
surfaced explicitly in the Live Alert Center and the Drift & Cold Start dashboard page.

## 18. Dashboard Modules

Executive Overview (fleet-wide KPIs and charts), Live Alert Center (filterable, ranked alert
table with full per-alert explainability), Entity Behavioral Profile (per-entity baseline and
deviation view), Threat Analytics (per-attack-type deep dives), Model Performance (metrics,
confusion matrix, feature importance), Drift & Cold Start (PSI monitor and cold-start entity
list), Attack Simulator (on-demand synthetic attack generation and live scoring), and Architecture
& About (system design, stack, limitations, and future scope).

## 19. Evaluation Metrics

See Sections 12-13 above and `outputs/evaluation_metrics.csv` / `outputs/model_comparison.csv` for
the complete, unedited set of metrics computed directly from model predictions on the held-out
chronological test split. No metric in this report or the dashboard has been hand-adjusted or
hard-coded.

## 20. Results

The Random Forest attack classifier achieves strong macro-level performance (macro F1 ~0.91)
despite the extreme imbalance, aided by class weighting. Anomaly detection is intentionally more
conservative: Isolation Forest catches roughly half of injected anomalies at a modest false-
positive rate, while comparison models (One-Class SVM, LOF) trade higher recall and precision for
slower, instance-based scoring. This is a realistic and expected outcome for unsupervised anomaly
detection on overlapping behavioral classes, not an artifact of a broken pipeline.

## 21. Feasibility

**Technical feasibility:** the full pipeline (data generation, training, and dashboard) runs
end-to-end in well under a minute on modest hardware using only open-source Python libraries.
**Operational feasibility:** the rule-based risk engine and explainability layer are simple enough
for a SOC analyst to audit and trust without a machine-learning background.

## 22. Challenges and Mitigation

| Challenge | Mitigation |
|---|---|
| Extreme class imbalance | Class-weighted classifiers; alert-budget-based evaluation instead of raw accuracy |
| False positives | Multi-signal risk scoring rather than a single threshold; explainability for triage |
| Concept drift | PSI monitoring; trusted-window baseline updates only |
| Cold-start entities | Global/entity-type fallback baselines; reduced alert confidence |
| Synthetic-to-real gap | Explicitly documented throughout; noise injected to avoid unrealistically clean separability |
| Real-time scalability | Millisecond-scale per-event scoring latency in this prototype; documented streaming extension path |

## 23. System Scalability

This MVP scores events in batch/on-demand fashion suitable for a hackathon demo. The documented
extension path is batch -> micro-batch -> true streaming (e.g., Kafka/Flink) with the same
feature-engineering and scoring logic wrapped behind a streaming consumer, since Isolation Forest
and Random Forest inference are both computationally cheap per event.

## 24. Privacy and Ethical Considerations

All data used in this prototype is synthetically generated; no real user, employee, or customer
data is used. In any real deployment, access-log analytics would need to comply with applicable
data-protection regulations, apply strict access controls to the analytics platform itself, and
ensure human review before any automated action (e.g., blocking an IP or revoking a session) is
taken against a real account.

## 25. Limitations

- Synthetic data only; real production traffic distributions and attacker behavior will differ.
- Risk-score weights are heuristic, not statistically fit against real labeled incidents.
- No true streaming infrastructure in this MVP.
- Rule-based rather than SHAP-based explainability.
- Not load-tested at enterprise data volumes.

## 26. Future Scope

Real streaming pipeline integration; SHAP-based explainability; analyst feedback loops for
continual retraining; graph-based lateral-movement detection; and validation against real,
anonymized SOC datasets before any production consideration.

## 27. References

- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest.* IEEE ICDM.
- Breunig, M. M., et al. (2000). *LOF: Identifying Density-Based Local Outliers.* ACM SIGMOD.
- Scholkopf, B., et al. (2001). *Estimating the Support of a High-Dimensional Distribution*
  (One-Class SVM). Neural Computation.
- Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1).
- scikit-learn documentation: https://scikit-learn.org/stable/
- Streamlit documentation: https://docs.streamlit.io/
- MITRE ATT&CK Framework (for attack-pattern reference): https://attack.mitre.org/

---

*This report describes a hackathon prototype built on synthetic data. Results should not be
interpreted as validated production performance.*
