"""
RQ1 Analysis: Performance Variation by Route, Time, Day, Location
===================================================================
Implements section 1.5.7 / Table 1.5.2 of the methodology chapter:

  Dimension     | Metric                          | Test
  --------------|----------------------------------|---------------------------
  Route         | OTP, avg delay, headway CoV      | Kruskal-Wallis
  Time of day   | metrics by AM/inter/PM/eve/night  | Kruskal-Wallis
  Day of week   | weekday vs weekend, per-day        | Kruskal-Wallis + Dunn's
  Location(stop)| delay/headway variance by stop     | ranked chronic black spots
                                                       (+ simple spatial summary;
                                                        true Moran's I needs stop
                                                        lat/lon - see note below)

USAGE
-----
    python rq1_analysis.py --data ../data/synthetic_trip_records.csv --outdir ../outputs

Swap --data to your real ../data/trip_records.csv once the collector has
gathered enough live data - no other changes needed, since both files share
the same schema.

OTP DEFINITION USED HERE
-------------------------
On-time = delay_min within [-1, +5] minutes (a common bus-sector OTP window,
adjust to your operator's own definition, e.g. TfL sometimes uses -2.5/+5).

NOTE ON SPATIAL ANALYSIS (Moran's I)
--------------------------------------
True Moran's I needs each stop's lat/lon and a spatial weights matrix. The
live collector currently logs naptanId/stationName but not coordinates.
TfL's StopPoint endpoint (/StopPoint/{naptanId}) returns lat/lon - add a
one-off lookup and merge on naptanId to enable full spatial autocorrelation;
this script instead ranks stops by delay to surface "black spots" so you
still get the location-based finding RQ1 needs.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal
import scikit_posthocs as sp  # for Dunn's test

sns.set_theme(style="whitegrid")

OTP_LOWER, OTP_UPPER = -1.0, 5.0  # minutes; adjust to your operator's OTP window


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["actual_arrival", "first_predicted_arrival"])
    df["on_time"] = df["delay_min"].between(OTP_LOWER, OTP_UPPER)
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=day_order, ordered=True)
    band_order = ["am_peak", "inter_peak", "pm_peak", "evening", "night"]
    df["time_band"] = pd.Categorical(df["time_band"], categories=band_order, ordered=True)
    return df


def summary_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    g = df.groupby(group_col, observed=True).agg(
        n_trips=("delay_min", "size"),
        mean_delay_min=("delay_min", "mean"),
        median_delay_min=("delay_min", "median"),
        std_delay_min=("delay_min", "std"),
        otp_pct=("on_time", lambda x: 100 * x.mean()),
        mean_headway_min=("headway_min", "mean"),
        headway_cov=("headway_min", lambda x: x.std() / x.mean() if x.mean() else np.nan),
    ).reset_index()
    return g.sort_values("otp_pct")


def kw_test(df: pd.DataFrame, group_col: str, value_col: str = "delay_min"):
    groups = [g[value_col].dropna().values for _, g in df.groupby(group_col, observed=True)]
    groups = [g for g in groups if len(g) > 0]
    stat, p = kruskal(*groups)
    return stat, p


def dunn_posthoc(df: pd.DataFrame, group_col: str, value_col: str = "delay_min") -> pd.DataFrame:
    return sp.posthoc_dunn(df, val_col=value_col, group_col=group_col, p_adjust="bonferroni")


def plot_box_by_route(df: pd.DataFrame, outdir: str):
    plt.figure(figsize=(9, 5.5))
    order = df.groupby("route")["delay_min"].median().sort_values().index
    sns.boxplot(data=df, x="route", y="delay_min", order=order, showfliers=False)
    plt.axhline(0, color="grey", linestyle="--", linewidth=1)
    plt.title("Delay Distribution by Route")
    plt.ylabel("Delay (minutes)")
    plt.xlabel("Route")
    plt.tight_layout()
    path = os.path.join(outdir, "rq1_box_delay_by_route.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_heatmap_time_day(df: pd.DataFrame, outdir: str):
    pivot = df.pivot_table(index="day_of_week", columns="time_band",
                            values="delay_min", aggfunc="mean", observed=True)
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r", center=0,
                cbar_kws={"label": "Mean delay (min)"})
    plt.title("Mean Delay by Day of Week and Time Band")
    plt.tight_layout()
    path = os.path.join(outdir, "rq1_heatmap_delay_time_day.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_otp_by_route_timeband(df: pd.DataFrame, outdir: str):
    pivot = df.pivot_table(index="route", columns="time_band",
                            values="on_time", aggfunc="mean", observed=True) * 100
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", vmin=50, vmax=100,
                cbar_kws={"label": "OTP (%)"})
    plt.title("On-Time Performance (%) by Route and Time Band")
    plt.tight_layout()
    path = os.path.join(outdir, "rq1_heatmap_otp_route_timeband.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_stop_blackspots(df: pd.DataFrame, outdir: str, top_n: int = 15):
    stop_summary = summary_table(df, "naptanId").sort_values("mean_delay_min", ascending=False).head(top_n)
    plt.figure(figsize=(9, 6))
    sns.barplot(data=stop_summary, y="naptanId", x="mean_delay_min",
                hue="naptanId", palette="Reds_r", legend=False)
    plt.title(f"Top {top_n} Highest-Delay Stops ('Black Spots')")
    plt.xlabel("Mean delay (minutes)")
    plt.ylabel("Stop (naptanId)")
    plt.tight_layout()
    path = os.path.join(outdir, "rq1_blackspot_stops.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")
    return stop_summary


def plot_weekday_vs_weekend(df: pd.DataFrame, outdir: str):
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df, x="is_weekend", y="delay_min", showfliers=False)
    plt.xticks([0, 1], ["Weekday", "Weekend"])
    plt.axhline(0, color="grey", linestyle="--", linewidth=1)
    plt.title("Delay: Weekday vs Weekend")
    plt.ylabel("Delay (minutes)")
    plt.tight_layout()
    path = os.path.join(outdir, "rq1_weekday_vs_weekend.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def run(data_path: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    df = load_data(data_path)
    print(f"Loaded {len(df)} trip records covering routes: {sorted(df['route'].unique())}\n")

    report_lines = []

    def log(msg):
        print(msg)
        report_lines.append(msg)

    # ---- 1. Overall performance ----
    log("=" * 70)
    log("OVERALL PERFORMANCE SUMMARY")
    log("=" * 70)
    log(f"Overall OTP: {100 * df['on_time'].mean():.1f}%")
    log(f"Overall mean delay: {df['delay_min'].mean():.2f} min "
        f"(median {df['delay_min'].median():.2f} min)")
    log(f"Overall mean headway: {df['headway_min'].mean():.2f} min\n")

    # ---- 2. By route ----
    log("=" * 70)
    log("BY ROUTE")
    log("=" * 70)
    route_summary = summary_table(df, "route")
    log(route_summary.to_string(index=False))
    stat, p = kw_test(df, "route")
    log(f"\nKruskal-Wallis (delay ~ route): H={stat:.2f}, p={p:.4g}")
    if p < 0.05:
        log("--> Statistically significant difference in delay across routes.")
        dunn = dunn_posthoc(df, "route")
        log("Dunn's post-hoc pairwise p-values (Bonferroni-adjusted):")
        log(dunn.round(4).to_string())
    plot_box_by_route(df, outdir)
    log("")

    # ---- 3. By time of day ----
    log("=" * 70)
    log("BY TIME OF DAY")
    log("=" * 70)
    band_summary = summary_table(df, "time_band")
    log(band_summary.to_string(index=False))
    stat, p = kw_test(df, "time_band")
    log(f"\nKruskal-Wallis (delay ~ time_band): H={stat:.2f}, p={p:.4g}")
    if p < 0.05:
        log("--> Statistically significant difference in delay across time bands.")
    log("")

    # ---- 4. By day of week ----
    log("=" * 70)
    log("BY DAY OF WEEK")
    log("=" * 70)
    day_summary = summary_table(df, "day_of_week")
    log(day_summary.to_string(index=False))
    stat, p = kw_test(df, "day_of_week")
    log(f"\nKruskal-Wallis (delay ~ day_of_week): H={stat:.2f}, p={p:.4g}")
    if p < 0.05:
        log("--> Statistically significant difference in delay across days.")
        dunn = dunn_posthoc(df, "day_of_week")
        log("Dunn's post-hoc pairwise p-values (Bonferroni-adjusted):")
        log(dunn.round(4).to_string())
    plot_weekday_vs_weekend(df, outdir)

    wknd_stat, wknd_p = stats.mannwhitneyu(
        df.loc[~df.is_weekend, "delay_min"], df.loc[df.is_weekend, "delay_min"]
    )
    log(f"\nMann-Whitney U (weekday vs weekend delay): U={wknd_stat:.0f}, p={wknd_p:.4g}\n")

    # ---- 5. By location (stop) ----
    log("=" * 70)
    log("BY LOCATION (STOP)")
    log("=" * 70)
    stop_bs = plot_stop_blackspots(df, outdir)
    log("Top 'black spot' stops by mean delay:")
    log(stop_bs[["naptanId", "n_trips", "mean_delay_min", "otp_pct"]].to_string(index=False))
    log("")

    # ---- 6. Combined heatmaps ----
    plot_heatmap_time_day(df, outdir)
    plot_otp_by_route_timeband(df, outdir)

    # ---- Save summary tables + text report ----
    route_summary.to_csv(os.path.join(outdir, "rq1_summary_by_route.csv"), index=False)
    band_summary.to_csv(os.path.join(outdir, "rq1_summary_by_timeband.csv"), index=False)
    day_summary.to_csv(os.path.join(outdir, "rq1_summary_by_day.csv"), index=False)
    stop_bs.to_csv(os.path.join(outdir, "rq1_summary_by_stop.csv"), index=False)

    report_path = os.path.join(outdir, "rq1_analysis_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nFull text report saved -> {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/synthetic_trip_records.csv")
    parser.add_argument("--outdir", default="../outputs")
    args = parser.parse_args()

    here = os.path.dirname(__file__)
    run(
        os.path.abspath(os.path.join(here, args.data)),
        os.path.abspath(os.path.join(here, args.outdir)),
    )


if __name__ == "__main__":
    main()
