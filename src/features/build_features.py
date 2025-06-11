import pandas as pd
from sklearn.model_selection import train_test_split
from src.features.transformations import (
    driver_distance_to_pickup,
    hour_of_day,
    build_driver_static_features,
    add_driver_static_features,
    is_weekend
)
from src.utils.store import AssignmentStore
from src.utils.config import load_config


def apply_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pipe(driver_distance_to_pickup)
          .pipe(hour_of_day)
          .pipe(is_weekend)
    )


def apply_static_features(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    driver_stats = build_driver_static_features(df_train, shuffle_features=True)
    df_train = add_driver_static_features(df_train, driver_stats)
    df_test = add_driver_static_features(df_test, driver_stats)
    return df_train, df_test, driver_stats


def main():
    store = AssignmentStore()
    config = load_config()

    # Load raw data and apply safe pre-split features
    df = store.get_processed("dataset.csv").sort_values("event_timestamp")
    df = apply_temporal_features(df)

    # Split for downstream modeling or processing
    df_train, df_test = train_test_split(df, test_size=config["test_size"], random_state=42, shuffle=False)

    # Add static features computed only on training data
    df_train, df_test, driver_stats = apply_static_features(df_train, df_test)

    # Save outputs
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    store.put_processed("transformed_dataset.csv", df_combined)
    store.put_processed("driver_static_features.csv", driver_stats)


if __name__ == "__main__":
    main()
