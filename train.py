"""Train an XGBoost model that predicts ETA to the next station."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "actual_next_station_eta_minutes"
DATE_COLUMN = "service_date"
GROUP_COLUMNS = ["train_number", DATE_COLUMN]

SOURCE_NUMERIC_COLUMNS = [
    "latitude",
    "longitude",
    "station_index",
    "route_station_count",
    "stations_remaining",
    "distance_from_source_km",
    "route_distance_km",
    "distance_remaining_km",
    "previous_segment_distance_km",
    "network_segment_distance_km",
    "segment_train_count",
    "scheduled_segment_minutes",
    "current_arrival_delay_minutes",
    "current_departure_delay_minutes",
    "previous_arrival_delay_minutes",
    "previous_departure_delay_minutes",
    "actual_segment_minutes",
    "historical_segment_speed_kmph",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "weather_code",
    "wind_speed_10m",
    "cloud_cover",
    "surface_pressure",
    "prediction_hour",
    "prediction_day_of_week",
    "prediction_month",
]

# The "next" fields below are scheduled/static upcoming-segment descriptions;
# they are not future actual observations.
FEATURES = [
    "latitude",
    "longitude",
    "station_index",
    "route_station_count",
    "stations_remaining",
    "distance_from_source_km",
    "route_distance_km",
    "distance_remaining_km",
    "next_segment_distance_km",
    "next_network_segment_distance_km",
    "next_segment_train_count",
    "scheduled_minutes_to_next",
    "current_arrival_delay_minutes",
    "current_departure_delay_minutes",
    "previous_arrival_delay_minutes",
    "previous_departure_delay_minutes",
    "actual_segment_minutes",
    "historical_segment_speed_kmph",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "weather_code",
    "wind_speed_10m",
    "cloud_cover",
    "surface_pressure",
    "prediction_hour",
    "prediction_day_of_week",
    "prediction_month",
]


def metrics(y_true: pd.Series, predictions: np.ndarray) -> dict:
    return {
        "mae_minutes": float(mean_absolute_error(y_true, predictions)),
        "rmse_minutes": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
        "rows": int(len(y_true)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/processed/final_eta_dataset.csv",
        help="Prepared historical ETA CSV.",
    )
    parser.add_argument("--output-dir", default="models/next_station_xgboost")
    parser.add_argument("--train-end", default="2024-09-20")
    parser.add_argument("--validation-end", default="2024-09-25")
    parser.add_argument("--test-end", default="2024-09-30")
    parser.add_argument("--estimators", type=int, default=700)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 150 trees for a fast pipeline check.",
    )
    return parser.parse_args()


def build_next_station_rows(data_path: Path) -> pd.DataFrame:
    required = list(
        dict.fromkeys(
            [
                "train_number",
                "station_code",
                "next_station_code",
                DATE_COLUMN,
                *SOURCE_NUMERIC_COLUMNS,
            ]
        )
    )
    data = pd.read_csv(
        data_path,
        usecols=required,
        dtype={
            "train_number": "string",
            "station_code": "string",
            "next_station_code": "string",
        },
        parse_dates=[DATE_COLUMN],
    )
    for column in SOURCE_NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.sort_values(
        ["train_number", DATE_COLUMN, "station_index"]
    ).reset_index(drop=True)
    journey = data.groupby(GROUP_COLUMNS, sort=False)

    # The following row's completed segment describes travel from the current
    # row to that following station, so these fields are shifted backward.
    data["next_row_station_index"] = journey["station_index"].shift(-1)
    data["following_station_code"] = journey["station_code"].shift(-1)
    data[TARGET] = journey["actual_segment_minutes"].shift(-1)
    data["next_segment_distance_km"] = journey[
        "previous_segment_distance_km"
    ].shift(-1)
    data["next_network_segment_distance_km"] = journey[
        "network_segment_distance_km"
    ].shift(-1)
    data["next_segment_train_count"] = journey["segment_train_count"].shift(-1)
    data["scheduled_minutes_to_next"] = journey["scheduled_segment_minutes"].shift(-1)

    valid = (
        data["next_row_station_index"].eq(data["station_index"] + 1)
        & data["following_station_code"].eq(data["next_station_code"])
        & data[TARGET].gt(0)
        & data["next_segment_distance_km"].gt(0)
    )
    data = data.loc[valid].copy()
    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = build_next_station_rows(Path(args.data))
    print(f"Usable next-station training rows: {len(data):,}")

    train_end = pd.Timestamp(args.train_end)
    validation_end = pd.Timestamp(args.validation_end)
    test_end = pd.Timestamp(args.test_end)
    train_mask = data[DATE_COLUMN].le(train_end)
    validation_mask = data[DATE_COLUMN].gt(train_end) & data[DATE_COLUMN].le(
        validation_end
    )
    test_mask = data[DATE_COLUMN].gt(validation_end) & data[DATE_COLUMN].le(test_end)
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError("One or more chronological splits are empty.")

    x_train, y_train = data.loc[train_mask, FEATURES], data.loc[train_mask, TARGET]
    x_validation = data.loc[validation_mask, FEATURES]
    y_validation = data.loc[validation_mask, TARGET]
    x_test, y_test = data.loc[test_mask, FEATURES], data.loc[test_mask, TARGET]

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        n_estimators=150 if args.quick else args.estimators,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=10,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        missing=np.nan,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_validation, y_validation)],
        verbose=25,
    )

    validation_predictions = np.maximum(model.predict(x_validation), 0)
    test_predictions = np.maximum(model.predict(x_test), 0)
    validation_errors = np.abs(y_validation.to_numpy() - validation_predictions)
    prediction_intervals = {
        "80": float(np.quantile(validation_errors, 0.80)),
        "90": float(np.quantile(validation_errors, 0.90)),
        "95": float(np.quantile(validation_errors, 0.95)),
    }
    scheduled_baseline = np.maximum(
        x_test["scheduled_minutes_to_next"].fillna(
            x_train["scheduled_minutes_to_next"].median()
        ),
        0,
    )

    result = {
        "model_type": "XGBRegressor",
        "objective": "ETA to the next station in minutes",
        "target": TARGET,
        "features": FEATURES,
        "split": {
            "train": f"<= {args.train_end}",
            "validation": f"{args.train_end} < date <= {args.validation_end}",
            "test": f"{args.validation_end} < date <= {args.test_end}",
        },
        "best_iteration": (
            int(model.best_iteration)
            if getattr(model, "best_iteration", None) is not None
            else None
        ),
        "validation": metrics(y_validation, validation_predictions),
        "test": metrics(y_test, test_predictions),
        "scheduled_next_station_baseline_test": metrics(y_test, scheduled_baseline),
    }

    model.save_model(output_dir / "eta_model.json")
    with open(output_dir / "feature_schema.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "target": TARGET,
                "features_in_order": FEATURES,
                "missing_value_policy": "Leave unavailable numeric values as NaN.",
                "prediction_unit": "minutes",
                "prediction_scope": "next station",
            },
            file,
            indent=2,
        )
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    with open(
        output_dir / "prediction_intervals.json", "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "method": "validation_absolute_error_quantiles",
                "intervals_minutes": prediction_intervals,
                "validation_rows": int(len(y_validation)),
            },
            file,
            indent=2,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()