"""
TfL Arrivals Collector
=======================
Polls the TfL Unified API (https://api.tfl.gov.uk) for real-time bus arrival
predictions on a chosen set of routes, and appends each poll as raw rows to
a CSV log.

WHY THIS EXISTS
---------------
TfL's API only exposes *real-time* predictions - there is no public endpoint
that hands you a ready-made historical delay/OTP dataset. To build the
dataset your dissertation's RQ1-RQ3 methodology needs (Table 1.5.1: "AVL
timestamps... route/stop IDs"), you have to poll the live feed repeatedly
over time and accumulate your own historical log. That is exactly what this
script does.

HOW TO RUN IT
-------------
One-off test:
    python tfl_collector.py --routes 24,38,73 --once

Continuous collection (recommended - run this in the background, or better,
schedule it with cron so it survives terminal closures):
    python tfl_collector.py --routes 24,38,73 --interval 60

Cron example (every minute, logging to a file):
    * * * * * cd /path/to/collector && /usr/bin/python3 tfl_collector.py --routes 24,38,73 --once >> collector.log 2>&1

GETTING AN API KEY (recommended, not strictly required)
--------------------------------------------------------
The API works without a key for low-volume/testing use (as used to build
this script), but TfL rate-limits unauthenticated requests. For sustained
multi-week collection, register a free key:
  1. https://api-portal.tfl.gov.uk/signup
  2. Sign in -> Products -> subscribe to the free 500-requests plan
  3. Profile -> copy your Primary key
  4. Set it as an environment variable: export TFL_APP_KEY=xxxxxxxx
     (the script will pick it up automatically)

OUTPUT
------
Appends to data/raw_snapshots.csv with columns:
    poll_time, lineId, vehicleId, tripId, naptanId, stationName,
    direction, destinationName, timeToStation, expectedArrival

Each row is one (vehicle, stop) prediction observed at one point in time.
Run build_trip_records.py afterwards to turn this raw log into clean
per-stop-visit delay/OTP records for analysis.
"""

import argparse
import csv
import os
import time
from datetime import datetime, timezone

import requests

BASE_URL = "https://api.tfl.gov.uk"
APP_KEY = os.environ.get("TFL_APP_KEY")  # optional but recommended

FIELDS = [
    "poll_time", "lineId", "vehicleId", "tripId", "naptanId",
    "stationName", "direction", "destinationName",
    "timeToStation", "expectedArrival",
]


def fetch_line_arrivals(line_id: str) -> list[dict]:
    """Fetch all real-time arrival predictions for a given bus route (line)."""
    url = f"{BASE_URL}/Line/{line_id}/Arrivals"
    params = {}
    if APP_KEY:
        params["app_key"] = APP_KEY
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def poll_once(routes: list[str], out_path: str) -> int:
    """Poll all given routes once and append results to the CSV log."""
    poll_time = datetime.now(timezone.utc).isoformat()
    rows_written = 0

    file_exists = os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()

        for line_id in routes:
            try:
                predictions = fetch_line_arrivals(line_id)
            except requests.RequestException as e:
                print(f"[{poll_time}] WARN: failed to fetch line {line_id}: {e}")
                continue

            for p in predictions:
                writer.writerow({
                    "poll_time": poll_time,
                    "lineId": p.get("lineId"),
                    "vehicleId": p.get("vehicleId"),
                    "tripId": p.get("tripId"),
                    "naptanId": p.get("naptanId"),
                    "stationName": p.get("stationName"),
                    "direction": p.get("direction"),
                    "destinationName": p.get("destinationName"),
                    "timeToStation": p.get("timeToStation"),
                    "expectedArrival": p.get("expectedArrival"),
                })
                rows_written += 1

    print(f"[{poll_time}] wrote {rows_written} prediction rows "
          f"for routes {routes} -> {out_path}")
    return rows_written


def main():
    parser = argparse.ArgumentParser(description="Poll TfL bus arrivals and log snapshots.")
    parser.add_argument("--routes", required=True,
                         help="Comma-separated bus route IDs, e.g. 24,38,73")
    parser.add_argument("--interval", type=int, default=60,
                         help="Seconds between polls (default: 60). Ignored with --once.")
    parser.add_argument("--once", action="store_true",
                         help="Poll a single time and exit (use this with cron).")
    parser.add_argument("--out", default="../data/raw_snapshots.csv",
                         help="Path to the output CSV log.")
    args = parser.parse_args()

    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if args.once:
        poll_once(routes, out_path)
        return

    print(f"Starting continuous collection for routes {routes}, "
          f"every {args.interval}s. Ctrl+C to stop.")
    try:
        while True:
            poll_once(routes, out_path)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
