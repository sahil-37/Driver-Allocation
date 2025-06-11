import pandas as pd
from src.utils.config import load_config
from src.utils.store import AssignmentStore
from src.utils.warnings import suppress_pandas_warnings


def main():
    store = AssignmentStore()
    config = load_config()

    # Load raw logs
    booking_df = store.get_raw("booking_log.csv")
    participant_df = store.get_raw("participant_log.csv")

    # Clean and pivot logs
    booking_df = clean_booking_log(booking_df)
    participant_df = clean_participant_log(participant_df)

    # Merge and label
    dataset = merge_flattened_logs(participant_df, booking_df)
    dataset = create_target(dataset, config["target"])
    dataset.rename(columns={"CREATED":"event_timestamp"}, inplace= True)

    # Save final dataset
    store.put_processed("dataset.csv", dataset)


@suppress_pandas_warnings
def clean_id_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64").astype(str)
    return df


def clean_booking_log(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce", utc=True)

    df = df.dropna(subset=["event_timestamp", "order_id", "driver_id"])
    df = clean_id_column(df, "order_id")
    df = clean_id_column(df, "driver_id")

    # Pivot booking statuses to columns
    booking_pivot = (
        df.pivot_table(
            index=["order_id", "driver_id", "trip_distance", "pickup_latitude", "pickup_longitude"],
            columns="booking_status",
            values="event_timestamp",
            aggfunc="first"
        )
        .reset_index()
    )

    # Drop columns
    booking_pivot.drop(columns=["CREATED", "DRIVER_FOUND"], errors="ignore", inplace=True)
    return booking_pivot


def clean_participant_log(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce", utc=True)

    df = df.dropna(subset=["event_timestamp", "order_id", "driver_id"])
    df = clean_id_column(df, "order_id")
    df = clean_id_column(df, "driver_id")

    # Pivot participant statuses to columns
    participant_pivot = (
        df.pivot_table(
            index=["order_id", "driver_id", "experiment_key", "driver_latitude", "driver_longitude", "driver_gps_accuracy"],
            columns="participant_status",
            values="event_timestamp",
            aggfunc="first"
        )
        .reset_index()
    )
    return participant_pivot


def merge_flattened_logs(participant_df: pd.DataFrame, booking_df: pd.DataFrame) -> pd.DataFrame:
    # Merge on order_id and driver_id
    merged_df = pd.merge(participant_df, booking_df, on=["order_id", "driver_id"], how="left")

    booking_lookup = (
        booking_df[["order_id", "pickup_latitude", "pickup_longitude"]]
        .drop_duplicates("order_id")
        .set_index("order_id")
    )

    # Fill missing pickup coordinates
    for col in ["pickup_latitude", "pickup_longitude"]:
        merged_df[col] = merged_df[col].fillna(
            merged_df["order_id"].map(booking_lookup[col])
        )

    # === Remove rows where customer cancelled BEFORE any driver response ===
    merged_df["driver_response_time"] = merged_df[["ACCEPTED", "REJECTED", "IGNORED"]].min(axis=1)
    merged_df["CUSTOMER_CANCELLED"] = pd.to_datetime(merged_df["CUSTOMER_CANCELLED"], errors="coerce")

    merged_df = merged_df[
        (merged_df["CUSTOMER_CANCELLED"].isna()) |
        (merged_df["CUSTOMER_CANCELLED"] > merged_df["driver_response_time"])
    ].drop(columns=["driver_response_time"])

    return merged_df



def create_target(df: pd.DataFrame, target_col: str = "is_accepted") -> pd.DataFrame:
    accepted = df["ACCEPTED"].notna()
    cancelled_after_accept = (
        df["DRIVER_CANCELLED"].notna() &
        (df["DRIVER_CANCELLED"] > df["ACCEPTED"])
    )

    df[target_col] = (accepted & ~cancelled_after_accept).astype(int)
    return df


if __name__ == "__main__":
    main()
