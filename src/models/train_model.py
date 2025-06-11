import toml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.features.feature_selection import compute_permutation_importance
from src.features.transformations import (
    build_driver_static_features,
    add_driver_static_features,
)
from src.models.classifier import SklearnClassifier
from src.utils.config import load_config
from src.utils.guardrails import validate_evaluation_metrics
from src.utils.store import AssignmentStore


@validate_evaluation_metrics
def main():
    store = AssignmentStore()
    config_path = "config.toml"
    config = load_config()

    # Load dataset
    df = store.get_processed("transformed_dataset.csv").sort_values("event_timestamp")
    target_column = config["target"]

    # Try all these features first
    candidate_features = [
        "driver_distance",
        "event_hour",
        "is_weekend",
        "driver_acceptance_rate",
        "driver_avg_daily_acceptances",
        "driver_avg_trip_distance_per_day",
        "driver_avg_ride_duration_per_day",
        "driver_completed_days",
    ]

    # Do a train/test split (keep time order)
    df_train, df_test = train_test_split(df, test_size=config["test_size"], random_state=42, shuffle=False)

    # Build driver stats from train data
    driver_stats = build_driver_static_features(df_train, shuffle_features=True)

    # Add these stats to train/test
    df_train = add_driver_static_features(df_train, driver_stats)
    df_test = add_driver_static_features(df_test, driver_stats)

    # Drop rows with missing values
    df_train = df_train[candidate_features + [target_column]].dropna()
    df_test = df_test[candidate_features + [target_column]].dropna()

    # Train initial model
    rf = RandomForestClassifier(**config["random_forest"])
    rf.fit(df_train[candidate_features], df_train[target_column])

    # Find important features
    importance_df = compute_permutation_importance(
        model=rf,
        X_test=df_test[candidate_features],
        y_test=df_test[target_column],
        scoring="precision",
        n_repeats=10,
        random_state=42
    )

    # Pick top K
    top_k = 5
    selected_features = importance_df["feature"].head(top_k).tolist()
    print("Selected top features based on permutation importance:", selected_features)

    # Save them to config
    config_data = toml.load(config_path)
    config_data["features"] = selected_features
    with open(config_path, "w") as f:
        toml.dump(config_data, f)
    print("Updated `features` in config.toml.")

    # Reload with new config
    config = load_config()
    final_features = config["features"]

    # Final model training
    rf_final = RandomForestClassifier(**config["random_forest"])
    model = SklearnClassifier(rf_final, final_features, target_column)
    model.train(df_train[final_features + [target_column]])

    # Evaluate and save
    metrics = model.evaluate(df_test[final_features + [target_column]])
    store.put_model("saved_model.pkl", model)
    store.put_metrics("metrics.json", metrics)
    store.put_processed("driver_static_features.csv", driver_stats)


if __name__ == "__main__":
    main()
