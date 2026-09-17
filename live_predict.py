"""Predict ETA to the next station using RailRadar, weather, and XGBoost."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from dotenv import load_dotenv

RAILRADAR_BASE_URL = "https://api.railradar.in/v1"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "weather_code",
    "wind_speed_10m",
    "cloud_cover",
    "surface_pressure",
]


def timestamp(value):
    return pd.Timestamp(value) if value else None


def minutes_between(later, earlier):
    later, earlier = timestamp(later), timestamp(earlier)
    if later is None or earlier is None:
        return np.nan
    return (later - earlier).total_seconds() / 60


def get_live_train(train_number: str, journey_date: str | None) -> dict:
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY is missing. Add it to your local .env file.")
    params = {"haltsOnly": "true", "includeCoordinates": "true"}
    if journey_date:
        params["date"] = journey_date
    response = requests.get(
        f"{RAILRADAR_BASE_URL}/trains/{train_number}/live",
        headers={"Authorization": f"Bearer {api_key}"},
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        message = payload.get("error", {}).get("message", "RailRadar request failed.")
        raise RuntimeError(message)
    return payload["data"]


def get_hourly_weather(latitude: float, longitude: float, at_time) -> dict:
    response = requests.get(
        OPEN_METEO_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(WEATHER_COLUMNS),
            "timezone": "Asia/Kolkata",
            "forecast_days": 1,
        },
        timeout=30,
    )
    response.raise_for_status()
    hourly = response.json()["hourly"]
    weather = pd.DataFrame(hourly)
    weather["time"] = pd.to_datetime(weather["time"])
    target_hour = pd.Timestamp(at_time).tz_localize(None).floor("h")
    closest = (weather["time"] - target_hour).abs().idxmin()
    return weather.loc[closest, WEATHER_COLUMNS].to_dict()


def load_edge(from_code: str, to_code: str, edges_path: Path) -> tuple:
    if not edges_path.exists():
        return np.nan, np.nan
    edges = pd.read_csv(edges_path)
    match = edges[
        (edges["from"].eq(from_code) & edges["to"].eq(to_code))
        | (edges["from"].eq(to_code) & edges["to"].eq(from_code))
    ]
    if match.empty:
        return np.nan, np.nan
    distance = float(match.iloc[0]["distance"])
    return (distance if distance > 0 else np.nan, float(match.iloc[0]["ntrains"]))


def select_current_and_next(live: dict):
    route = sorted(live["route"], key=lambda stop: int(stop["sequence"]))
    live_time = timestamp(live.get("lastUpdatedAt"))
    if live_time is None:
        live_time = pd.Timestamp.now(tz="Asia/Kolkata")
    elif live_time.tzinfo is None:
        live_time = live_time.tz_localize("Asia/Kolkata")

    # RailRadar's nextHalt identifies the target station for this prediction.
    next_halt = live.get("nextHalt") or {}
    next_code = next_halt.get("stationCode")
    next_sequence = next_halt.get("sequence")
    next_stop = next(
        (
            stop
            for stop in route
            if (next_sequence is not None and stop["sequence"] == next_sequence)
            or (next_code and stop["stationCode"] == next_code)
        ),
        None,
    )

    # Reject future projected departures by requiring departure <= API update.
    departed = []
    for stop in route:
        departure = timestamp(stop.get("actualDeparture"))
        if departure is None:
            continue
        if departure.tzinfo is None:
            departure = departure.tz_localize("Asia/Kolkata")
        if departure <= live_time + pd.Timedelta(minutes=2):
            if next_stop is None or int(stop["sequence"]) < int(next_stop["sequence"]):
                departed.append(stop)

    if not departed:
        raise RuntimeError("No completed departure was found before the next halt.")
    current = max(departed, key=lambda stop: int(stop["sequence"]))

    if next_stop is None:
        current_index = route.index(current)
        if current_index + 1 >= len(route):
            raise RuntimeError("The train has reached its final station.")
        next_stop = route[current_index + 1]

    current_index = route.index(current)
    next_index = route.index(next_stop)
    if next_index != current_index + 1:
        raise RuntimeError(
            "RailRadar route alignment is inconsistent: nextHalt is not "
            "immediately after the latest completed departure."
        )
    previous = route[current_index - 1] if current_index else None
    return route, previous, current, next_stop, live_time


def make_live_features(live: dict, edges_path: Path):
    route, previous, current, next_stop, live_time = select_current_and_next(live)
    prediction_time = timestamp(current["actualDeparture"])
    if prediction_time.tzinfo is None:
        prediction_time = prediction_time.tz_localize("Asia/Kolkata")

    current_lat, current_lng = current.get("lat"), current.get("lng")
    next_lat, next_lng = next_stop.get("lat"), next_stop.get("lng")
    if current_lat is None or current_lng is None:
        raise RuntimeError("RailRadar did not return current-station coordinates.")

    weather_lat, weather_lng = float(current_lat), float(current_lng)
    progress = live.get("currentLocation", {}).get("segmentProgress")
    if progress is not None and next_lat is not None and next_lng is not None:
        progress = min(max(float(progress), 0.0), 1.0)
        weather_lat += progress * (float(next_lat) - weather_lat)
        weather_lng += progress * (float(next_lng) - weather_lng)
    weather = get_hourly_weather(weather_lat, weather_lng, live_time)

    previous_segment_distance = (
        float(current["distance"]) - float(previous["distance"]) if previous else np.nan
    )
    actual_segment = (
        minutes_between(current.get("actualArrival"), previous.get("actualDeparture"))
        if previous
        else np.nan
    )
    historical_speed = (
        previous_segment_distance / (actual_segment / 60)
        if previous_segment_distance > 0 and actual_segment > 0
        else np.nan
    )
    if pd.notna(historical_speed) and historical_speed > 200:
        historical_speed = np.nan

    next_segment_distance = float(next_stop["distance"]) - float(current["distance"])
    network_distance, next_train_count = load_edge(
        current["stationCode"], next_stop["stationCode"], edges_path
    )
    departure_delay = current.get("delayDeparture")
    if departure_delay is None:
        departure_delay = live.get("delayMinutes")

    features = {
        "latitude": float(current_lat),
        "longitude": float(current_lng),
        "station_index": route.index(current) + 1,
        "route_station_count": len(route),
        "stations_remaining": len(route) - route.index(current) - 1,
        "distance_from_source_km": float(current["distance"]),
        "route_distance_km": float(live["train"]["distance"]),
        "distance_remaining_km": float(live["train"]["distance"])
        - float(current["distance"]),
        "next_segment_distance_km": next_segment_distance,
        "next_network_segment_distance_km": network_distance,
        "next_segment_train_count": next_train_count,
        "scheduled_minutes_to_next": minutes_between(
            next_stop.get("scheduledArrival"), current.get("scheduledDeparture")
        ),
        "current_arrival_delay_minutes": current.get("delayArrival"),
        "current_departure_delay_minutes": departure_delay,
        "previous_arrival_delay_minutes": previous.get("delayArrival")
        if previous
        else np.nan,
        "previous_departure_delay_minutes": previous.get("delayDeparture")
        if previous
        else np.nan,
        "actual_segment_minutes": actual_segment,
        "historical_segment_speed_kmph": historical_speed,
        **weather,
        "prediction_hour": prediction_time.hour,
        "prediction_day_of_week": prediction_time.dayofweek,
        "prediction_month": prediction_time.month,
    }
    return features, current, next_stop, live_time, prediction_time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_number", help="5-digit train number, e.g. 12919")
    parser.add_argument("--date", help="Journey start date: YYYY-MM-DD")
    parser.add_argument("--model-dir", default="models/next_station_xgboost")
    parser.add_argument("--edges", default="data/raw/IRN_edges.csv")
    args = parser.parse_args()

    load_dotenv()
    train_number = args.train_number.strip().zfill(5)
    model_dir = Path(args.model_dir)
    with open(model_dir / "feature_schema.json", encoding="utf-8") as file:
        feature_names = json.load(file)["features_in_order"]

    live = get_live_train(train_number, args.date)
    values, current, next_stop, live_time, prediction_time = make_live_features(
        live, Path(args.edges)
    )
    matrix = pd.DataFrame([values]).reindex(columns=feature_names)

    model = xgb.XGBRegressor()
    model.load_model(model_dir / "eta_model.json")
    full_segment_eta = max(float(model.predict(matrix)[0]), 0)
    elapsed = max((live_time - prediction_time).total_seconds() / 60, 0)
    remaining_eta = max(full_segment_eta - elapsed, 0)
    arrival = live_time + pd.Timedelta(minutes=remaining_eta)

    print(
        f"Current/last departed station: {current['stationCode']} — "
        f"{current['stationName']}"
    )
    print(f"Next station: {next_stop['stationCode']} — {next_stop['stationName']}")
    print(f"Current delay: {live.get('delayMinutes')} minutes")
    print(
        f"Train {train_number} — predicted arrival at "
        f"{next_stop['stationCode']} in {round(remaining_eta)} minutes"
    )
    print(f"Predicted arrival time: {arrival.strftime('%Y-%m-%d %H:%M %Z')}")

    interval_file = model_dir / "prediction_intervals.json"
    if interval_file.exists():
        with open(interval_file, encoding="utf-8") as file:
            calibration = json.load(file)
        level = "90"
        margin = float(calibration["intervals_minutes"][level])
        lower = max(remaining_eta - margin, 0)
        upper = remaining_eta + margin
        lower_time = live_time + pd.Timedelta(minutes=lower)
        upper_time = live_time + pd.Timedelta(minutes=upper)
        print(f"{level}% calibrated ETA interval: {round(lower)}–{round(upper)} minutes")
        print(
            f"Expected arrival window: {lower_time.strftime('%Y-%m-%d %H:%M')} "
            f"to {upper_time.strftime('%Y-%m-%d %H:%M')}"
        )
    else:
        print("Prediction interval unavailable; run train.py first.")


if __name__ == "__main__":
    main()