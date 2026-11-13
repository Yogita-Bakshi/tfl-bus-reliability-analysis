"""
RQ2 Analysis: Factors Associated with Poor Reliability
==========================================================
Implements section 1.5.8 of the methodology:
  1. Bivariate screening (chi-square / correlation) of candidate factors
  2. Multiple logistic regression: on_time ~ factors (with VIF check)
  3. Random Forest feature-importance ranking (non-linear cross-check)

FACTORS USED
------------
From trip_records.csv directly: route, time_band, day_of_week, is_weekend,
headway_min.
Optionally joined from weather.csv (if present, via fetch_weather.py):
temperature_c, precipitation_mm, windspeed_kmh, weathercode.
If weather.csv is not found, the script proceeds without it and prints a
note - this is a legitimate documented scope reduction, not an error.

USAGE
-----
    python rq2_analysis.py --data trip_records.csv --weather weather.csv --outdir .
    python rq2_analysis.py --data trip_records.csv --outdir .     # without weather

OTP DEFINITION: same as RQ1 (delay_min within [-1, 5] minutes = on time).
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set_theme(style="whitegrid")
OTP_LOWER, OTP_UPPER = -1.0, 5.0


def load_data(data_path: str, weather_path: str | None) -> pd.DataFrame:
    df = pd.read_csv(data_path, parse_dates=["actual_arrival"])
    df["on_time"] = df["delay_min"].between(OTP_LOWER, OTP_UPPER).astype(int)
    df["poor_reliability"] = 1 - df["on_time"]

    df["route"] = df["route"].astype(str)  # route is a categorical ID, not a magnitude

    if weather_path and os.path.exists(weather_path):
        weather = pd.read_csv(weather_path, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["actual_arrival"].dt.date)
        df = df.merge(weather, on=["date", "hour"], how="left")
        print(f"Joined weather data: {weather_path}")
    else:
        print("No weather.csv found - proceeding with operational/contextual "
              "factors only (route, time_band, day_of_week, headway). "
              "Run fetch_weather.py first to add weather as a factor.")

    return df


def bivariate_screening(df: pd.DataFrame, factors: list[str], report: list[str]):
    report.append("=" * 70)
    report.append("BIVARIATE SCREENING (chi-square for categorical factors)")
    report.append("=" * 70)
    for factor in factors:
        if not pd.api.types.is_numeric_dtype(df[factor]):
            table = pd.crosstab(df[factor], df["poor_reliability"])
            chi2, p, dof, _ = chi2_contingency(table)
            report.append(f"{factor}: chi2={chi2:.2f}, dof={dof}, p={p:.4g}"
                          f"{'  --> significant' if p < 0.05 else ''}")
        else:
            valid = df[[factor, "poor_reliability"]].dropna()
            if len(valid) > 1 and valid[factor].std() > 0:
                r, p = pointbiserialr(valid["poor_reliability"], valid[factor])
                report.append(f"{factor}: point-biserial r={r:.3f}, p={p:.4g}"
                              f"{'  --> significant' if p < 0.05 else ''}")
    report.append("")


def vif_check(X: pd.DataFrame, report: list[str]):
    report.append("Variance Inflation Factors (VIF > 5 suggests multicollinearity):")
    vif_data = pd.DataFrame({
        "feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    })
    report.append(vif_data.round(2).to_string(index=False))
    report.append("")


def logistic_regression(df: pd.DataFrame, feature_cols: list[str], report: list[str]):
    report.append("=" * 70)
    report.append("MULTIPLE LOGISTIC REGRESSION: poor_reliability ~ factors")
    report.append("=" * 70)

    model_df = df.dropna(subset=feature_cols + ["poor_reliability"]).copy()
    X = pd.get_dummies(model_df[feature_cols], drop_first=True).astype(float)
    X = sm.add_constant(X)
    y = model_df["poor_reliability"].astype(float)

    vif_check(X.drop(columns=["const"]), report)

    model = sm.Logit(y, X).fit(disp=False)
    report.append(model.summary().as_text())
    report.append("")
    report.append("Odds ratios (exp(coef)):")
    report.append(np.exp(model.params).round(3).to_string())
    report.append("")
    return model


def random_forest_importance(df: pd.DataFrame, feature_cols: list[str], outdir: str, report: list[str]):
    report.append("=" * 70)
    report.append("RANDOM FOREST FEATURE IMPORTANCE (non-linear cross-check)")
    report.append("=" * 70)

    model_df = df.dropna(subset=feature_cols + ["poor_reliability"]).copy()
    X = model_df[feature_cols].copy()
    encoders = {}
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
    y = model_df["poor_reliability"]

    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, class_weight="balanced")
    rf.fit(X, y)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    report.append(importance.to_string(index=False))
    report.append("")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance, y="feature", x="importance", hue="feature",
                palette="viridis", legend=False)
    plt.title("Random Forest Feature Importance: Predicting Poor Reliability")
    plt.xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(outdir, "rq2_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

    return importance


def run(data_path: str, weather_path: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    df = load_data(data_path, weather_path)
    print(f"Loaded {len(df)} trip records\n")

    report = []
    base_factors = ["route", "time_band", "day_of_week", "headway_min"]
    weather_factors = ["temperature_c", "precipitation_mm", "windspeed_kmh"]
    available_weather = [f for f in weather_factors if f in df.columns]
    feature_cols = base_factors + available_weather

    bivariate_screening(df, feature_cols, report)
    logistic_regression(df, feature_cols, report)
    random_forest_importance(df, feature_cols, outdir, report)

    for line in report:
        print(line)

    report_path = os.path.join(outdir, "rq2_analysis_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"\nFull text report saved -> {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="trip_records.csv")
    parser.add_argument("--weather", default="weather.csv")
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()
    run(args.data, args.weather, args.outdir)


if __name__ == "__main__":
    main()
