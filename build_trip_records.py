"""
Build Trip Records
===================
Converts the raw snapshot log produced by tfl_collector.py into a clean,
one-row-per-stop-visit performance dataset suitable for the RQ1 descriptive
/comparative analysis (Table 1.5.2 of the methodology).

METHODOLOGICAL NOTE ON "DELAY"
-------------------------------
TfL's real-time Arrivals API does not return a scheduled time directly - it
only ever returns a live, continuously-updated *predicted* arrival time. So
"delay" here is built as a proxy, in line with how AVL-only studies in the
literature (e.g. Moreira-Matias et al., 2015) handle the same constraint:

  - first_predicted_arrival : the EARLIEST expectedArrival TfL gave for this
                               vehicle approaching this stop (i.e. the
                               prediction made when the bus was furthest away
                               / first entered the polling horizon).
  - actual_arrival           : the LAST expectedArrival TfL gave before the
                               vehicle's timeToStation reached its minimum
                               (i.e. TfL's own best real-time estimate of
                               when the bus actually arrived).
  - delay_min                : (actual_arrival - first_predicted_arrival) in
                               minutes. A positive value means the vehicle
                               arrived later than originally predicted when
                               first detected; this captures in-transit
                               deterioration (traffic, dwell time, bunching)
                               even though it is not "vs scheduled timetable"
                               delay in the strict Kittelson (2003) sense.

TO GET TRUE SCHEDULE-BASED OTP: fetch /Line/{id}/Timetable/{fromStopId} once
per route (static, changes rarely) and join on tripId/naptanId to replace
first_predicted_arrival with the actual scheduled time. That endpoint's
response format is more complex (nested interval/schedule arrays) and is
left as a follow-up enhancement - flag this in your Limitations section if
you use the proxy as-is, or ask me to build that join next.

OUTPUT
------
data/trip_records.csv, columns:
    route, naptanId, stationName, direction, vehicleId, tripId,
    first_predicted_arrival, actual_arrival, delay_min,
    date, day_of_week, is_weekend, hour, time_band, headway_min
"""

import argparse
import os

import numpy as np
import pandas as pd

TIME_BANDS = [
    (0, 6, "night"),
    (6, 10, "am_peak"),
    (10, 16, "inter_peak"),
    (16, 19, "pm_peak"),
    (19, 22, "evening"),
    (22, 24, "night"),
]


def time_band(hour: int) -> str:
    for start, end, label in TIME_BANDS:
        if start <= hour < end:
            return label
    return "night"


def build_trip_records(raw_path: str, out_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path, parse_dates=["poll_time", "expectedArrival"])
    df = df.dropna(subset=["vehicleId", "naptanId", "lineId"])

    # Each (line, vehicle, trip, stop) combination is one "stop visit" that
    # TfL predicted repeatedly as the bus approached.
    group_cols = ["lineId", "vehicleId", "tripId", "naptanId"]
    df = df.sort_values("poll_time")

    records = []
    for keys, g in df.groupby(group_cols):
        line_id, vehicle_id, trip_id, naptan_id = keys
        g = g.sort_values("poll_time")

        first_predicted = g.iloc[0]["expectedArrival"]
        closest_idx = g["timeToStation"].idxmin()
        closest_row = g.loc[closest_idx]
        actual_arrival = closest_row["expectedArrival"]
        station_name = closest_row["stationName"]
        direction = closest_row["direction"]

        delay_min = (actual_arrival - first_predicted).total_seconds() / 60.0

        records.append({
            "route": line_id,
            "naptanId": naptan_id,
            "stationName": station_name,
            "direction": direction,
            "vehicleId": vehicle_id,
            "tripId": trip_id,
            "first_predicted_arrival": first_predicted,
            "actual_arrival": actual_arrival,
            "delay_min": delay_min,
        })

    trip_df = pd.DataFrame(records)
    if trip_df.empty:
        print("No trip records could be built - is raw_snapshots.csv populated yet?")
        return trip_df

    trip_df["date"] = trip_df["actual_arrival"].dt.date
    trip_df["day_of_week"] = trip_df["actual_arrival"].dt.day_name()
    trip_df["is_weekend"] = trip_df["actual_arrival"].dt.dayofweek >= 5
    trip_df["hour"] = trip_df["actual_arrival"].dt.hour
    trip_df["time_band"] = trip_df["hour"].apply(time_band)

    # Headway: gap between consecutive vehicles at the same stop/route/direction
    trip_df = trip_df.sort_values(["route", "naptanId", "direction", "actual_arrival"])
    trip_df["headway_min"] = (
        trip_df.groupby(["route", "naptanId", "direction"])["actual_arrival"]
        .diff().dt.total_seconds() / 60.0
    )

    trip_df.to_csv(out_path, index=False)
    print(f"Wrote {len(trip_df)} trip records -> {out_path}")
    return trip_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="../data/raw_snapshots.csv")
    parser.add_argument("--out", default="../data/trip_records.csv")
    args = parser.parse_args()

    here = os.path.dirname(__file__)
    build_trip_records(
        os.path.abspath(os.path.join(here, args.raw)),
        os.path.abspath(os.path.join(here, args.out)),
    )


if __name__ == "__main__":
    main()
