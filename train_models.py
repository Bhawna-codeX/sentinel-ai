"""
train_models.py
SentinelAI - trains the two-stage hybrid pipeline:
  Stage A: unsupervised anomaly detection (Isolation Forest, primary;
           One-Class SVM and LOF as comparison baselines)
  Stage B: supervised attack-type classifier (Random Forest, primary;
           HistGradientBoosting and Logistic Regression as comparisons)

Also produces evaluation outputs (metrics, confusion matrix, feature
importance, anomaly score distribution) required by the dashboard.

Run:
    python train_models.py
"""

import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    confusion_matrix, classification_report, roc_auc_score,
)

from utils import (
    TRAIN_CSV, TEST_CSV, MODEL_DIR, OUTPUT_DIR, FEATURE_COLUMNS,
    build_label_encoders, engineer_features, get_feature_matrix,
)

warnings.filterwarnings("ignore")
SEED = 42


def load_and_prepare():
    train_raw = pd.read_csv(TRAIN_CSV, parse_dates=["timestamp"])
    test_raw = pd.read_csv(TEST_CSV, parse_dates=["timestamp"])

    train_feat = engineer_features(train_raw)
    test_feat = engineer_features(test_raw)

    encoders = build_label_encoders(pd.concat([train_raw, test_raw], ignore_index=True))

    X_train = get_feature_matrix(train_feat, encoders)
    X_test = get_feature_matrix(test_feat, encoders)

    y_train_bin = train_raw["anomaly_label"].values
    y_test_bin = test_raw["anomaly_label"].values
    y_train_multi = train_raw["attack_type"].values
    y_test_multi = test_raw["attack_type"].values

    return X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi, encoders, train_feat, test_feat


def top_k_precision(y_true, scores, k_frac=0.01):
    """Precision within a top-k% alert budget (a common SOC operating constraint)."""
    n = len(scores)
    k = max(1, int(n * k_frac))
    top_idx = np.argsort(scores)[-k:]
    return precision_score(y_true[top_idx], np.ones(k), zero_division=0), k


def train_stage_a(X_train, X_test, y_train_bin, y_test_bin, scaler):
    Xtr_s = scaler.transform(X_train)
    Xte_s = scaler.transform(X_test)

    results = {}

    # 1. Isolation Forest (primary)
    contamination = max(0.01, min(0.2, y_train_bin.mean()))
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=SEED, n_jobs=-1)
    t0 = time.time()
    iso.fit(Xtr_s)
    iso_latency = (time.time() - t0) / max(len(Xtr_s), 1) * 1000  # ms/event (fit, rough proxy)
    iso_scores_raw = -iso.score_samples(Xte_s)  # higher = more anomalous
    iso_scores_norm = (iso_scores_raw - iso_scores_raw.min()) / (iso_scores_raw.max() - iso_scores_raw.min() + 1e-9)
    iso_pred = (iso.predict(Xte_s) == -1).astype(int)

    results["isolation_forest"] = _score_binary(y_test_bin, iso_pred, iso_scores_norm)
    results["isolation_forest"]["latency_ms_per_event"] = round(iso_latency, 4)

    # 2. One-Class SVM (comparison baseline, trained on a subsample for speed)
    sub_n = min(4000, len(Xtr_s))
    sub_idx = np.random.RandomState(SEED).choice(len(Xtr_s), sub_n, replace=False)
    ocsvm = OneClassSVM(kernel="rbf", nu=max(0.01, min(0.3, contamination)), gamma="scale")
    ocsvm.fit(Xtr_s[sub_idx])
    ocsvm_pred = (ocsvm.predict(Xte_s) == -1).astype(int)
    ocsvm_scores_raw = -ocsvm.decision_function(Xte_s)
    ocsvm_scores_norm = (ocsvm_scores_raw - ocsvm_scores_raw.min()) / (ocsvm_scores_raw.max() - ocsvm_scores_raw.min() + 1e-9)
    results["one_class_svm"] = _score_binary(y_test_bin, ocsvm_pred, ocsvm_scores_norm)

    # 3. Local Outlier Factor (comparison baseline, novelty mode)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)
    lof.fit(Xtr_s[sub_idx])
    lof_pred = (lof.predict(Xte_s) == -1).astype(int)
    lof_scores_raw = -lof.decision_function(Xte_s)
    lof_scores_norm = (lof_scores_raw - lof_scores_raw.min()) / (lof_scores_raw.max() - lof_scores_raw.min() + 1e-9)
    results["local_outlier_factor"] = _score_binary(y_test_bin, lof_pred, lof_scores_norm)

    return iso, results, iso_scores_norm


def _score_binary(y_true, y_pred, scores_norm):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, scores_norm)
    fpr = ((y_pred == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)
    top1_prec, k = top_k_precision(y_true, scores_norm, 0.01)
    return {
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "pr_auc": round(pr_auc, 4), "false_positive_rate": round(fpr, 4),
        "top1pct_alert_budget_precision": round(top1_prec, 4), "alert_budget_k": k,
    }


def train_stage_b(X_train, X_test, y_train_multi, y_test_multi):
    """Train attack-type classifiers on ALL events (multi-class incl. 'normal')."""
    results = {}
    models = {}

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=14, class_weight="balanced",
        random_state=SEED, n_jobs=-1,
    )
    rf.fit(X_train, y_train_multi)
    models["random_forest"] = rf
    results["random_forest"] = _score_multiclass(y_test_multi, rf.predict(X_test))

    hgb = HistGradientBoostingClassifier(max_depth=8, max_iter=200, random_state=SEED)
    hgb.fit(X_train, y_train_multi)
    models["hist_gradient_boosting"] = hgb
    results["hist_gradient_boosting"] = _score_multiclass(y_test_multi, hgb.predict(X_test))

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train, y_train_multi)
    models["logistic_regression"] = logreg
    results["logistic_regression"] = _score_multiclass(y_test_multi, logreg.predict(X_test))

    # Select best by macro F1
    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    return models[best_name], best_name, results


def _score_multiclass(y_true, y_pred):
    return {
        "accuracy": round((y_true == y_pred).mean(), 4),
        "macro_precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def main():
    np.random.seed(SEED)
    print("Loading data and engineering features ...")
    (X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi,
     encoders, train_feat, test_feat) = load_and_prepare()

    scaler = StandardScaler()
    scaler.fit(X_train)

    print("Training Stage A (anomaly detection models) ...")
    iso_model, stage_a_results, iso_scores_norm = train_stage_a(X_train, X_test, y_train_bin, y_test_bin, scaler)
    print(json.dumps(stage_a_results, indent=2))

    print("Training Stage B (attack-type classifiers) ...")
    best_clf, best_clf_name, stage_b_results = train_stage_b(X_train, X_test, y_train_multi, y_test_multi)
    print(f"Best classifier: {best_clf_name}")
    print(json.dumps(stage_b_results, indent=2))

    # ---------------- Persist model artifacts ----------------
    joblib.dump(iso_model, os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    joblib.dump(best_clf, os.path.join(MODEL_DIR, "attack_classifier.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
    joblib.dump(FEATURE_COLUMNS, os.path.join(MODEL_DIR, "feature_columns.pkl"))
    joblib.dump({"name": best_clf_name}, os.path.join(MODEL_DIR, "best_classifier_meta.pkl"))

    # ---------------- Evaluation outputs ----------------
    eval_rows = []
    for name, m in stage_a_results.items():
        row = {"stage": "anomaly_detection", "model": name}
        row.update(m)
        eval_rows.append(row)
    for name, m in stage_b_results.items():
        row = {"stage": "attack_classification", "model": name}
        row.update(m)
        eval_rows.append(row)
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_metrics.csv"), index=False)

    model_comparison = pd.DataFrame([
        {"stage": "anomaly_detection", "model": k, **v} for k, v in stage_a_results.items()
    ])
    model_comparison.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

    # Confusion matrix for best classifier
    y_pred_best = best_clf.predict(X_test)
    labels = sorted(pd.unique(np.concatenate([y_test_multi, y_pred_best])))
    cm = confusion_matrix(y_test_multi, y_pred_best, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {best_clf_name} (synthetic test data)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=130, facecolor="white")
    plt.close()

    # Per-class classification report
    report = classification_report(y_test_multi, y_pred_best, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(os.path.join(OUTPUT_DIR, "per_class_report.csv"))

    # Attack distribution chart
    raw = pd.read_csv(TRAIN_CSV)
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = raw["attack_type"].value_counts()
    ax.bar(counts.index, counts.values, color="#3fa9f5")
    ax.set_yscale("log")
    ax.set_title("Attack-Type Distribution (training data, log scale)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "attack_distribution.png"), dpi=130, facecolor="white")
    plt.close()

    # Anomaly score distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(iso_scores_norm[y_test_bin == 0], bins=40, alpha=0.6, label="Normal", color="#3fa9f5")
    ax.hist(iso_scores_norm[y_test_bin == 1], bins=40, alpha=0.6, label="Anomalous", color="#e05263")
    ax.set_title("Isolation Forest Anomaly Score Distribution (test data)")
    ax.set_xlabel("Normalized anomaly score"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "anomaly_score_distribution.png"), dpi=130, facecolor="white")
    plt.close()

    # Feature importance (from Random Forest if selected, else train one quickly for reference)
    if best_clf_name == "random_forest":
        importances = best_clf.feature_importances_
    else:
        rf_ref = RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=-1)
        rf_ref.fit(X_train, y_train_multi)
        importances = rf_ref.feature_importances_
    imp_series = pd.Series(importances, index=FEATURE_COLUMNS).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(imp_series.index[-15:], imp_series.values[-15:], color="#3fa9f5")
    ax.set_title("Top 15 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=130, facecolor="white")
    plt.close()

    print("Training complete. Model artifacts saved to /models, evaluation outputs saved to /outputs.")


if __name__ == "__main__":
    main()
