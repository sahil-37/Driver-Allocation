from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


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
        # Some classifiers might not have predict_proba, so check
        if hasattr(self.clf, "predict_proba"):
            y_proba = self.clf.predict_proba(X_test)[:, 1]

        metrics = {}

        metrics["accuracy"] = accuracy_score(y_test, y_pred)
        metrics["precision"] = precision_score(y_test, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_test, y_pred, zero_division=0)
        metrics["f1_score"] = f1_score(y_test, y_pred, zero_division=0)
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist() 
        #balanced metrics
        metrics["balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred)
        metrics["balanced_precision"] = precision_score(y_test, y_pred, zero_division=0, average='macro')
        metrics["balanced_recall"] = recall_score(y_test, y_pred, zero_division=0, average='macro')
        metrics["balanced_f1_score"] = f1_score(y_test, y_pred, zero_division=0, average='macro')

        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

        return metrics

    def predict(self, df: pd.DataFrame):
        return self.clf.predict_proba(df[self.features])[:, 1]
