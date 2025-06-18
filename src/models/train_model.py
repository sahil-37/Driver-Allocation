import toml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.features.transformations import build_driver_static_features, add_driver_static_features
from src.models.classifier import SklearnClassifier
from src.utils.config import load_config
from src.utils.store import AssignmentStore
from src.utils.guardrails import validate_evaluation_metrics


@validate_evaluation_metrics
def main():
    store = AssignmentStore()
    config = load_config()
    target = config["target"]
    config_path = "config.toml"

    # Load and sort the dataset
    df = store.get_processed("transformed_dataset.csv").sort_values("event_timestamp")

    # Initial train/test split
    df_train_full, df_test = train_test_split(df, test_size=config["test_size"], shuffle=False)

    # Split df_train_full into train/validation
    df_train, df_val = train_test_split(df_train_full, test_size=0.2, shuffle=False)

    # Build static features from training only
    driver_stats = build_driver_static_features(df_train, shuffle_features=True)

    # Apply to train/val/test
    df_train = add_driver_static_features(df_train, driver_stats)
    df_val   = add_driver_static_features(df_val, driver_stats)
    df_test  = add_driver_static_features(df_test, driver_stats)

    # Candidate features (all available)
    candidate_features = [
        "driver_distance", "event_hour", "is_weekend",
        "driver_acceptance_rate", "driver_avg_daily_acceptances",
        "driver_avg_ride_duration_per_day",
        "driver_completed_days"
    ]

    # Drop missing rows
    df_train = df_train[candidate_features + [target]].dropna()
    df_val   = df_val[candidate_features + [target]].dropna()
    df_test  = df_test[candidate_features + [target]].dropna()

    # Step 1: Train model on all features
    rf = RandomForestClassifier(**config["random_forest"])
    rf.fit(df_train[candidate_features], df_train[target])

    # Step 2: Get feature importances
    importances = pd.Series(rf.feature_importances_, index=candidate_features)
    sorted_features = importances.sort_values(ascending=False)

    # Step 3-4: Try different top-k feature sets and evaluate
    results = []
    for k in range(2, 6):
        top_k_features = sorted_features.head(k).index.tolist()
        model = SklearnClassifier(RandomForestClassifier(**config["random_forest"]), top_k_features, target)
        model.train(df_train[top_k_features + [target]])
        metrics = model.evaluate(df_val[top_k_features + [target]])
        metrics["num_features"] = k
        results.append((top_k_features, metrics))

    # Step 5: Choose best performing based on F1 (or another)
    best_result = min(results, key=lambda x: x[1]["fpr"])
    best_features, best_metrics = best_result
    print(f"Selected features: {best_features}")
    print(f"Validation fpr: {best_metrics['fpr']:.4f}")

    # Final model on train + val
    df_full_train = pd.concat([df_train, df_val], ignore_index=True)
    final_model = SklearnClassifier(RandomForestClassifier(**config["random_forest"]), best_features, target)
    final_model.train(df_full_train[best_features + [target]])

    # Evaluate on test set
    test_metrics = final_model.evaluate(df_test[best_features + [target]])
    # Save outputs
    config_data = toml.load(config_path)
    config_data["features"] = best_features
    with open(config_path, "w") as f:
        toml.dump(config_data, f)

    store.put_model("saved_model.pkl", final_model)
    store.put_metrics("metrics.json", test_metrics)
    store.put_processed("driver_static_features.csv", driver_stats)


if __name__ == "__main__":
    main()
