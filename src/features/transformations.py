import pandas as pd
from haversine import haversine
from src.utils.time import robust_hour_of_iso_date
import numpy as np

pd.set_option('future.no_silent_downcasting', True)

# Calculates haversine distance between driver and pickup location
def driver_distance_to_pickup(df: pd.DataFrame) -> pd.DataFrame:
    df["driver_distance"] = df.apply(
        lambda r: haversine(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ),
        axis=1,
    )
    return df

# Extracts the hour of the event (0-23)
def hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    df["event_hour"] = df["event_timestamp"].apply(robust_hour_of_iso_date)
    return df

# Marks whether the event happened on a weekend (Saturday or Sunday)
def is_weekend(df: pd.DataFrame) -> pd.DataFrame:
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce", utc=True)
    df["is_weekend"] = (df["event_timestamp"].dt.weekday >= 5).astype(int)
    return df

# Builds driver-level static features from training data
def build_driver_static_features(df_train: pd.DataFrame, shuffle_features: bool = True) -> pd.DataFrame:
    df = df_train.copy()

    # Ensure datetime fields are correctly parsed
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce", utc=True)
    df["ACCEPTED"] = pd.to_datetime(df["ACCEPTED"], errors="coerce", utc=True)
    df["COMPLETED"] = pd.to_datetime(df["COMPLETED"], errors="coerce", utc=True)
    df["trip_distance"] = pd.to_numeric(df["trip_distance"], errors="coerce")

    # Create flag for whether driver accepted
    df["is_accepted"] = df["ACCEPTED"].notna().astype(int)

    # Extract the offer date (used to compute daily aggregates)
    df["offer_date"] = df["event_timestamp"].dt.date

    # Aggregate driver acceptance stats
    general_stats = (
        df.groupby("driver_id")
        .agg(
            driver_total_accepted=("is_accepted", "sum"),
            driver_total_offers=("order_id", "count"),
            active_days=("offer_date", "nunique"),
        )
        .reset_index()
    )

    # Compute acceptance rate and average acceptances per active day
    general_stats["driver_acceptance_rate"] = (
        general_stats["driver_total_accepted"] / general_stats["driver_total_offers"]
    )
    general_stats["driver_avg_daily_acceptances"] = (
        general_stats["driver_total_accepted"] / general_stats["active_days"].replace(0, np.nan)
    )

    # Filter only completed rides to build ride-related stats
    completed_df = df[df["COMPLETED"].notna()].copy()
    completed_df["pickup_distance_km"] = completed_df.apply(
        lambda r: haversine(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ),
        axis=1,
    )
    completed_df["total_trip_distance_km"] = completed_df["pickup_distance_km"] + completed_df["trip_distance"]
    completed_df["ride_duration_sec"] = (completed_df["COMPLETED"] - completed_df["ACCEPTED"]).dt.total_seconds()
    completed_df["ride_date"] = completed_df["COMPLETED"].dt.date

    # Aggregate ride metrics per driver per day
    daily_means = (
        completed_df.groupby(["driver_id", "ride_date"])
        .agg(
            avg_trip_distance_day=("total_trip_distance_km", "mean"),
            avg_ride_duration_day=("ride_duration_sec", "mean"),
        )
        .reset_index()
    )

    # Average those daily stats across all days for each driver
    daily_stats = (
        daily_means.groupby("driver_id")
        .agg(
            driver_avg_trip_distance_per_day=("avg_trip_distance_day", "mean"),
            driver_avg_ride_duration_per_day=("avg_ride_duration_day", "mean"),
            driver_completed_days=("ride_date", "nunique"),
        )
        .reset_index()
    )

    # Merge general and ride-related stats
    final_stats = pd.merge(general_stats, daily_stats, on="driver_id", how="left").fillna(0)

    # Optional: shuffle static features to break direct identity linkage (avoids leakage)
    if shuffle_features:
        features_to_shuffle = [
            "driver_total_accepted",
            "driver_total_offers",
            "active_days",
            "driver_acceptance_rate",
            "driver_avg_daily_acceptances",
            "driver_avg_trip_distance_per_day",
            "driver_avg_ride_duration_per_day",
            "driver_completed_days",
        ]
        features_to_shuffle = [col for col in features_to_shuffle if col in final_stats.columns]

        for feature_col in features_to_shuffle:
            final_stats[feature_col] = np.random.permutation(final_stats[feature_col].values)
        print("Static driver features have been shuffled.")

    return final_stats

# Merge the static features with any given dataset
def add_driver_static_features(df: pd.DataFrame, *stats: pd.DataFrame) -> pd.DataFrame:
    static_columns = [
        "driver_total_accepted", "active_days", "driver_total_offers",
        "driver_acceptance_rate", "driver_avg_daily_acceptances",
        "driver_avg_trip_distance_per_day", "driver_avg_ride_duration_per_day",
        "driver_completed_days"
    ]

    # Remove old static columns if they exist
    df = df.drop(columns=[col for col in static_columns if col in df.columns], errors="ignore")

    # Merge with provided stats
    for stat in stats:
        df = df.merge(stat, on="driver_id", how="left")

    return df.fillna(0)
