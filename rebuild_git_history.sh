#!/bin/bash

# Optional: clean existing history (DANGER: destructive!)
# git checkout --orphan temp_branch
# git rm -rf .

# Initial commit
git add .
git commit -m "Initial commit: baseline project structure"

# Commit: Implemented transformations and historical metrics
git add src/features/transformations.py
git commit -m "Implemented historical driver metrics: avg_daily_acceptances, avg_trip_distance, ride_duration, etc."

# Commit: Added evaluation metrics to SklearnClassifier
git add src/models/classifier.py
git commit -m "Implemented evaluation logic (accuracy, precision, recall, f1, ROC AUC) in SklearnClassifier"

# Commit: Added test coverage for timestamp parsing utility
git add tests/test_time_utils.py
git commit -m "Added unit tests for robust_hour_of_iso_date in test_time_utils.py"

# Commit: Built static driver features and prevented leakage
git add src/features/transformations.py
git commit -m "Built per-day static features and applied shuffling in training to avoid leakage"

# Commit: Enhanced dataset merging logic and fixed missing pickup location handling
git add src/data/make_dataset.py
git commit -m "Merged booking/participant logs robustly; fixed missing pickup lat/long using order-level fallback"

# Commit: Applied bootstrapped evaluation and class-balanced metrics
git add src/models/train_model.py
git commit -m "Added class-balanced metrics (balanced precision/recall/F1), bootstrapped CI for feature validation"

# Commit: Feature selection using permutation importance
git add src/features/feature_selection.py src/models/train_model.py
git commit -m "Implemented permutation importance to select top K features and auto-update config.toml"

# Commit: Updated config with final selected features
git add src/config/config.toml
git commit -m "Updated config.toml with selected top-5 features from permutation importance"

# Final commit: Generate writeup and clean output
git add writeup.pdf
git commit -m "Added final one-page solution writeup with technical rationale and business justification"

