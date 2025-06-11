import pandas as pd
from src.utils.config import load_config
from src.utils.store import AssignmentStore
from src.utils.warnings import suppress_pandas_warnings

# Main function to build the dataset
def main():
    store = AssignmentStore()
    config = load_config()

    # Load both booking and participant logs
    booking_df = store.get_raw("booking_log.csv")
    participant_df = store.get_raw("participant_log.csv")

    # Clean and reshape the data
    booking_df = clean_booking_log(booking_df)
    participant_df = clean_participant_log(participant_df)

    # Combine both logs into one dataset and create target variable
    dataset = merge_flattened_logs(participant_df, booking_df)
    dataset = create_target(dataset, config["target"])

    # Rename the 'CREATED' column to 'event_timestamp' for clarity
    dataset.rename(columns={"CREATED": "event_timestamp"}, inplace=True)

    # Save the final dataset
    store.put_processed("dataset.csv", dataset)

# Helper to make sure ID columns are all clean strings
@suppress_pandas_warnings
def clean_id_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64").astype(str)
    return df

# Clean booking log and reshape it
def clean_booking_log(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert event_timestamp to datetime
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce", utc=True)

    # Drop any rows missing important info
    df = df.dropna(subset=["event_timestamp", "order_id", "driver_id"])

    # Clean up order_id and driver_id
    df = clean_id_column(df, "order_id")
    df = clean_id_column(df, "driver_id")

    # Make each booking status a column (pivot the data)
    booking_pivot = (
        df.pivot_table(
            index=["order_id", "driver_id", "trip_distance", "pickup_latitude", "pickup_longitude"],
            columns="booking_status",
            values="event_timestamp",
            aggfunc="first"
        )
        .reset_index()
    )

    # Drop unused status columns
    booking_pivot.drop(columns=["CREATED", "DRIVER_FOUND"], errors="ignore", inplace=True)
    return booking_pivot

# Clean participant log and reshape it
def clean_participant_log(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse event_timestamp
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce", utc=True)

    # Drop rows missing IDs or timestamps
    df = df.dropna(subset=["event_timestamp", "order_id", "driver_id"])

    # Clean IDs
    df = clean_id_column(df, "order_id")
    df = clean_id_column(df, "driver_id")

    # Pivot so each participant status is its own column
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

# Combine both logs together, carefully
def merge_flattened_logs(participant_df: pd.DataFrame, booking_df: pd.DataFrame) -> pd.DataFrame:
    # Merge both tables on order_id and driver_id
    merged_df = pd.merge(participant_df, booking_df, on=["order_id", "driver_id"], how="left")

    # Create a quick lookup to fill in missing pickup coordinates later
    booking_lookup = (
        booking_df[["order_id", "pickup_latitude", "pickup_longitude"]]
        .drop_duplicates("order_id")
        .set_index("order_id")
    )

    # Fill in any missing pickup locations from the lookup
    for col in ["pickup_latitude", "pickup_longitude"]:
        merged_df[col] = merged_df[col].fillna(
            merged_df["order_id"].map(booking_lookup[col])
        )

    # Remove rows where the customer cancelled before the driver did anything
    merged_df["driver_response_time"] = merged_df[["ACCEPTED", "REJECTED", "IGNORED"]].min(axis=1)
    merged_df["CUSTOMER_CANCELLED"] = pd.to_datetime(merged_df["CUSTOMER_CANCELLED"], errors="coerce")

    # Only keep rows where the customer cancelled AFTER the driver acted (or never cancelled)
    merged_df = merged_df[
        (merged_df["CUSTOMER_CANCELLED"].isna()) |
        (merged_df["CUSTOMER_CANCELLED"] > merged_df["driver_response_time"])
    ].drop(columns=["driver_response_time"])

    return merged_df

# Create binary label for training the model
def create_target(df: pd.DataFrame, target_col: str = "is_accepted") -> pd.DataFrame:
    # Positive class if ACCEPTED is not null
    accepted = df["ACCEPTED"].notna()

    # But reject if driver cancelled AFTER accepting
    cancelled_after_accept = (
        df["DRIVER_CANCELLED"].notna() &
        (df["DRIVER_CANCELLED"] > df["ACCEPTED"])
    )

    df[target_col] = (accepted & ~cancelled_after_accept).astype(int)
    return df

# Run the main function if this script is executed
if __name__ == "__main__":
    main()
