"""
Fetch Historical Weather (London)
===================================
Pulls free historical hourly weather data for London from the Open-Meteo
Historical Weather API (no API key required, CC BY 4.0 licence) and saves
it to data/weather.csv so RQ2's analysis can join weather as one of the
contextual factors in Table 1.5.3 of the methodology.

USAGE
-----
    python fetch_weather.py --start 2026-07-17 --end 2026-07-24

Match --start/--end to the date range your raw_snapshots.csv actually
covers (check with the date column, or just use the dates you were
collecting data).

OUTPUT
------
data/weather.csv, columns: date, hour, temperature_c, precipitation_mm,
windspeed_kmh, weathercode
"""

import argparse
import os

import pandas as pd
import requests

LATITUDE, LONGITUDE = 51.5074, -0.1278  # Central London
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,precipitation,windspeed_10m,weathercode",
        "timezone": "Europe/London",
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "temperature_c": hourly["temperature_2m"],
        "precipitation_mm": hourly["precipitation"],
        "windspeed_kmh": hourly["windspeed_10m"],
        "weathercode": hourly["weathercode"],
    })
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    return df.drop(columns=["datetime"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out", default="weather.csv")
    args = parser.parse_args()

    df = fetch_weather(args.start, args.end)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} hourly weather rows -> {args.out}")


if __name__ == "__main__":
    main()
