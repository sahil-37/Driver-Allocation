# src/features/feature_selection.py

import pandas as pd
from sklearn.inspection import permutation_importance

def compute_permutation_importance(model, X_test, y_test, scoring='f1_score', n_repeats=10, random_state=42):
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        scoring=scoring,
        random_state=random_state
    )

    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values(by='importance_mean', ascending=False).reset_index(drop=True)

    return importance_df
