"""
Generate Synthetic Trip Records
=================================
Creates a synthetic dataset with the SAME SCHEMA that build_trip_records.py
produces from real TfL data, so you can develop/test/demo the RQ1 analysis
pipeline right now, then swap in the real trip_records.csv once your
collector has gathered a few weeks of live data (just point rq1_analysis.py
at the real file - no code changes needed).

The synthetic data deliberately encodes realistic patterns so the
statistical tests in rq1_analysis.py have something meaningful to find:
  - Route 73 is the least reliable (bunching-prone, high traffic corridor)
  - AM/PM peaks are worse than inter-peak/night
  - Weekdays are worse than weekends
  - A handful of stops are chronic "black spots" with elevated delay
    regardless of route/time (mimics a junction/congestion pinch point)
"""

import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

ROUTES = {
    "24": {"base_delay": 1.0, "noise": 2.5, "stops": [f"2400{i}" for i in range(1, 13)]},
    "38": {"base_delay": 1.8, "noise": 3.0, "stops": [f"3800{i}" for i in range(1, 15)]},
    "73": {"base_delay": 3.2, "noise": 4.2, "stops": [f"7300{i}" for i in range(1, 14)]},  # least reliable
    "159": {"base_delay": 0.6, "noise": 1.8, "stops": [f"1590{i}" for i in range(1, 11)]},  # most reliable
}

# A handful of chronic "black spot" stops that add extra delay regardless of route/time
BLACK_SPOTS = {"73005", "73009", "38007"}

TIME_BAND_EFFECT = {
    "am_peak": 2.5, "pm_peak": 3.0, "inter_peak": 0.5,
    "evening": 0.2, "night": -0.8,
}
HOUR_TO_BAND = {**{h: "night" for h in range(0, 6)},
                **{h: "am_peak" for h in range(6, 10)},
                **{h: "inter_peak" for h in range(10, 16)},
                **{h: "pm_peak" for h in range(16, 19)},
                **{h: "evening" for h in range(19, 22)},
                **{h: "night" for h in range(22, 24)}}

DAY_EFFECT = {
    "Monday": 1.2, "Tuesday": 0.8, "Wednesday": 0.8, "Thursday": 1.0,
    "Friday": 1.8, "Saturday": -0.5, "Sunday": -1.0,
}

DIRECTIONS = ["inbound", "outbound"]


def generate(n_days: int = 28, trips_per_stop_per_day: int = 14, out_path: str = None):
    rows = []
    start_date = pd.Timestamp("2026-06-15")

    for day_offset in range(n_days):
        date = start_date + pd.Timedelta(days=day_offset)
        day_name = date.day_name()
        day_eff = DAY_EFFECT[day_name]

        for route, cfg in ROUTES.items():
            for stop in cfg["stops"]:
                black_spot_eff = 2.5 if stop in BLACK_SPOTS else 0.0
                for direction in DIRECTIONS:
                    for _ in range(trips_per_stop_per_day):
                        hour = int(RNG.choice(range(5, 24),
                                               p=_hour_weights()))
                        band = HOUR_TO_BAND[hour]
                        minute = int(RNG.integers(0, 60))

                        actual_arrival = pd.Timestamp(
                            year=date.year, month=date.month, day=date.day,
                            hour=hour, minute=minute
                        )

                        mean_delay = (
                            cfg["base_delay"] + TIME_BAND_EFFECT[band]
                            + day_eff + black_spot_eff
                        )
                        delay_min = RNG.normal(mean_delay, cfg["noise"])
                        first_predicted = actual_arrival - pd.Timedelta(minutes=delay_min)

                        rows.append({
                            "route": route,
                            "naptanId": stop,
                            "stationName": f"Stop {stop}",
                            "direction": direction,
                            "vehicleId": f"SIM{RNG.integers(1000, 9999)}",
                            "tripId": f"T{RNG.integers(100000, 999999)}",
                            "first_predicted_arrival": first_predicted,
                            "actual_arrival": actual_arrival,
                            "delay_min": round(delay_min, 2),
                            "date": date.date(),
                            "day_of_week": day_name,
                            "is_weekend": day_name in ("Saturday", "Sunday"),
                            "hour": hour,
                            "time_band": band,
                        })

    trip_df = pd.DataFrame(rows)
    trip_df = trip_df.sort_values(["route", "naptanId", "direction", "actual_arrival"])
    trip_df["headway_min"] = (
        trip_df.groupby(["route", "naptanId", "direction"])["actual_arrival"]
        .diff().dt.total_seconds() / 60.0
    )

    if out_path:
        trip_df.to_csv(out_path, index=False)
        print(f"Wrote {len(trip_df)} synthetic trip records -> {out_path}")
    return trip_df


def _hour_weights():
    """Rough diurnal ridership curve: more trips at peak hours."""
    hours = list(range(5, 24))
    weights = np.array([
        1, 3, 6, 8, 5, 3, 2, 2, 2, 3, 4,     # 5-15
        5, 7, 8, 6, 3, 2, 2, 1               # 16-23
    ], dtype=float)
    weights = weights[:len(hours)]
    return weights / weights.sum()


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    out = os.path.abspath(os.path.join(here, "../data/synthetic_trip_records.csv"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    generate(out_path=out)
