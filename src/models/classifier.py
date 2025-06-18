from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import recall_score, confusion_matrix, average_precision_score, classification_report, precision_score, f1_score


class Classifier(ABC):
    @abstractmethod
    def train(self, *params) -> None:
        pass

    @abstractmethod
    def evaluate(self, *params) -> Dict[str, float]:
        pass

    @abstractmethod
    def predict(self, *params) -> np.ndarray:
        pass


class SklearnClassifier(Classifier):
    def __init__(
        self, estimator: BaseEstimator, features: List[str], target: str,
    ):
        self.clf = estimator
        self.features = features
        self.target = target

    def train(self, df_train: pd.DataFrame):
        self.clf.fit(df_train[self.features], df_train[self.target])

    def evaluate(self, df_test: pd.DataFrame) -> dict:
        X_test = df_test[self.features]
        y_test = df_test[self.target]

        y_pred = self.clf.predict(X_test)
        y_proba = None
        
        
        if hasattr(self.clf, "predict_proba"):
            y_proba = self.clf.predict_proba(X_test)[:, 1]

        metrics = {}

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        metrics = {
            "precision": precision_score(y_test,y_pred),
            "recall": recall_score(y_test,y_pred),
            "f1_score": f1_score(y_test,y_pred),
            "confusion_matrix":    [[int(tn), int(fp)], [int(fn), int(tp)]],
            "fpr":                 fp / (fp + tn) if (fp + tn) else 0.0,   # False-Positive Rate
            "specificity":         tn / (tn + fp) if (tn + fp) else 0.0,   # True-Negative Rate
        }
        metrics["classification_report"] = classification_report(y_test,y_pred, output_dict=True, zero_division=0)

        if y_proba is not None:
            metrics["pr_auc"]   = average_precision_score(y_test, y_proba)  # area under PR curve
        return metrics

    def predict(self, df: pd.DataFrame):
        return self.clf.predict_proba(df[self.features])[:, 1]
