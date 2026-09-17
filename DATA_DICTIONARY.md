# Final ETA dataset: data dictionary

## Row grain

One row is one historical prediction point immediately after a train's actual
departure from a non-terminal station.

## Next-station training target

`train.py` derives `actual_next_station_eta_minutes` by shifting the following
row's `actual_segment_minutes` backward within each
`train_number` + `service_date` journey. Therefore, row i uses only features
known at the current station and its target is actual travel time from row i's
departure to row i+1's arrival.

Rows are retained only when the following station index is exactly
`station_index + 1`, its code matches `next_station_code`, the target is
positive, and the upcoming segment distance is positive.

## Identifiers and route position

- `train_number`, `train_name`, `service_date`
- `station_code`, `station_name`, `zone`
- `station_index`, `route_station_count`, `stations_remaining`
- `previous_station_code`, `next_station_code`
- `distance_from_source_km`, `route_distance_km`,
  `distance_remaining_km`

## Segment and schedule features

- `previous_segment_distance_km`: route-distance difference for the completed
  segment.
- `network_segment_distance_km`, `segment_train_count`: values from the
  undirected IRN edge table.
- `scheduled_arrival_timestamp`, `scheduled_departure_timestamp`
- `prediction_timestamp`: scheduled departure plus the recorded departure
  delay.
- `scheduled_segment_minutes`, `scheduled_remaining_minutes`
- `scheduled_eta_from_prediction_minutes`: destination's scheduled arrival
  minus the actual prediction timestamp. It can be negative when the train is
  already later than the destination's scheduled arrival.

For next-station training, the following row's static/scheduled segment fields
are shifted onto the current row as:

- `next_segment_distance_km`
- `next_network_segment_distance_km`
- `next_segment_train_count`
- `scheduled_minutes_to_next`

No future actual delay or arrival value is used as an input feature. The
following station's completed actual segment time is used only as the target.

## Delay and completed-segment features

- `current_arrival_delay_minutes`, `current_departure_delay_minutes`
- `previous_arrival_delay_minutes`, `previous_departure_delay_minutes`
- `actual_segment_minutes`: actual arrival at the current station minus actual
  departure from the previous station; nonpositive source inconsistencies are
  left empty.
- `historical_segment_speed_kmph`: completed segment distance divided by
  completed segment time. Inconsistent derived values above 200 km/h are left
  empty, not imputed.

## Coordinates and weather

- `latitude`, `longitude`, `coordinate_source`
- `weather_hour`
- `temperature_2m`, `relative_humidity_2m`, `precipitation`, `rain`
- `weather_code`, `wind_speed_10m`, `cloud_cover`, `surface_pressure`
- `weather_source`

Weather is matched to the current station and the hour containing
`prediction_timestamp`. Open-Meteo requests use a 0.1-degree grid and
`Asia/Kolkata` local time. Exact station codes without a verified coordinate
retain empty coordinate and weather fields.

## Calendar features

- `prediction_hour`
- `prediction_day_of_week`: Monday = 0 through Sunday = 6
- `prediction_month`

## Important training notes

- Read `train_number` as a string to preserve leading zeroes.
- Use a chronological split based on `service_date`, not a random row split.
- The trained model predicts one current-station row at a time and returns ETA
  to that row's next station.
- Do not treat identifiers, timestamps, or source-label columns as numeric
  features without explicit encoding.
- Empty values are genuine unavailable/invalid-source values; no synthetic
  imputations were written into the CSV.