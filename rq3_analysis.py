"""
RQ3 Analysis: Predictive Analytics for Forecasting Disruption
==================================================================
Implements section 1.5.9 / Table 1.5.4 of the methodology (scoped down to
what's feasible within a dissertation timeline - binary classification
only; multi-class severity and full time-series forecasting are noted as
future work rather than attempted here, since they need substantially more
historical data than a short collection window provides).

APPROACH
--------
  1. Chronological split (NOT random) into train/test, to avoid leakage
     and reflect a realistic forecasting scenario (train on earlier data,
     predict later data).
  2. Three models compared: Logistic Regression (baseline), Random Forest,
     Gradient Boosting.
  3. Naive baseline: always predict the majority class (usually "on time"),
     to demonstrate genuine predictive value beyond chance.
  4. Evaluation: accuracy, precision, recall, F1, AUC-ROC, confusion matrix.
  5. Feature importance retained for interpretability, linking back to RQ2.

USAGE
-----
    python rq3_analysis.py --data trip_records.csv --outdir .

NOTE ON DATA VOLUME
--------------------
Predictive models need enough historical depth to learn real patterns.
With only a few days of collection, results should be read as a
proof-of-concept / methodology validation rather than a production-ready
forecaster - state this explicitly in your Limitations section.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.preprocessing import LabelEncoder

sns.set_theme(style="whitegrid")
OTP_LOWER, OTP_UPPER = -1.0, 5.0


def load_and_prepare(data_path: str):
    df = pd.read_csv(data_path, parse_dates=["actual_arrival"])
    df["on_time"] = df["delay_min"].between(OTP_LOWER, OTP_UPPER).astype(int)
    df = df.sort_values("actual_arrival").reset_index(drop=True)

    feature_cols = ["route", "time_band", "day_of_week", "headway_min", "hour"]
    feature_cols = [c for c in feature_cols if c in df.columns]
    df = df.dropna(subset=feature_cols + ["on_time"]).reset_index(drop=True)

    X = df[feature_cols].copy()
    X["route"] = X["route"].astype(str)  # route is a categorical ID, not a magnitude
    encoders = {}
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
    y = df["on_time"]

    return df, X, y, feature_cols


def chronological_split(X: pd.DataFrame, y: pd.Series, train_frac: float = 0.7, val_frac: float = 0.15):
    n = len(X)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return (
        X.iloc[:train_end], y.iloc[:train_end],
        X.iloc[train_end:val_end], y.iloc[train_end:val_end],
        X.iloc[val_end:], y.iloc[val_end:],
    )


def evaluate(name: str, y_true, y_pred, y_proba, report: list[str]):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = float("nan")  # only one class present in y_true

    report.append(f"\n--- {name} ---")
    report.append(f"Accuracy:  {acc:.3f}")
    report.append(f"Precision: {prec:.3f}")
    report.append(f"Recall:    {rec:.3f}")
    report.append(f"F1-score:  {f1:.3f}")
    report.append(f"AUC-ROC:   {auc:.3f}")
    cm = confusion_matrix(y_true, y_pred)
    report.append(f"Confusion matrix (rows=actual, cols=predicted):\n{cm}")
    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


def run(data_path: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    df, X, y, feature_cols = load_and_prepare(data_path)
    print(f"Loaded {len(df)} trip records; features used: {feature_cols}\n")

    report = []
    report.append("=" * 70)
    report.append("RQ3: PREDICTIVE MODELLING - ON-TIME vs DELAYED CLASSIFICATION")
    report.append("=" * 70)
    report.append(f"Total records: {len(df)}")
    report.append(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}\n")

    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(X, y)
    report.append(f"Train: {len(X_train)}  |  Validation: {len(X_val)}  |  Test: {len(X_test)}\n")

    results = []

    # Naive baseline: always predict the majority class from training data
    majority_class = y_train.mode()[0]
    naive_preds = np.full(len(y_test), majority_class)
    naive_proba = np.full(len(y_test), y_train.mean())
    results.append(evaluate("Naive baseline (majority class)", y_test, naive_preds, naive_proba, report))

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    results.append(evaluate("Logistic Regression", y_test, lr_preds, lr_proba, report))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    results.append(evaluate("Random Forest", y_test, rf_preds, rf_proba, report))

    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_test)
    gb_proba = gb.predict_proba(X_test)[:, 1]
    results.append(evaluate("Gradient Boosting", y_test, gb_preds, gb_proba, report))

    # Summary comparison table
    results_df = pd.DataFrame(results)
    report.append("\n" + "=" * 70)
    report.append("MODEL COMPARISON SUMMARY")
    report.append("=" * 70)
    report.append(results_df.round(3).to_string(index=False))
    results_df.to_csv(os.path.join(outdir, "rq3_model_comparison.csv"), index=False)

    # Feature importance chart (from the best tree-based model, Gradient Boosting)
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": gb.feature_importances_,
    }).sort_values("importance", ascending=False)
    report.append("\nGradient Boosting feature importance:")
    report.append(importance.to_string(index=False))

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance, y="feature", x="importance", hue="feature",
                palette="mako", legend=False)
    plt.title("Feature Importance (Gradient Boosting) - Predicting On-Time Arrival")
    plt.tight_layout()
    path = os.path.join(outdir, "rq3_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

    # Model comparison bar chart
    plt.figure(figsize=(8, 5))
    plot_df = results_df.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1", "auc"],
                               var_name="metric", value_name="score")
    sns.barplot(data=plot_df, x="metric", y="score", hue="model")
    plt.title("Model Comparison Across Evaluation Metrics")
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    path = os.path.join(outdir, "rq3_model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

    for line in report:
        print(line)

    report_path = os.path.join(outdir, "rq3_analysis_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"\nFull text report saved -> {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="trip_records.csv")
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()
    run(args.data, args.outdir)


if __name__ == "__main__":
    main()
