import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

class FeatureSelector:
    """
    Dimensionality reduction and feature selection engine supporting Variance Threshold,
    Mutual Information, and Random Forest Feature Importance.
    """

    def __init__(self, method: str = 'rf_importance', n_features: int = 10):
        self.method = method
        self.n_features = n_features
        self.selected_features: List[str] = []
        self.feature_scores: Dict[str, float] = {}

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
        """
        Perform feature selection and return reduced DataFrame with feature importance scores.
        """
        n_features_to_select = min(self.n_features, X.shape[1])

        if self.method == 'variance':
            selector = VarianceThreshold(threshold=0.01)
            X_reduced = selector.fit_transform(X)
            selected_indices = np.where(selector.get_support())[0]
            self.selected_features = list(X.columns[selected_indices])
            variances = selector.variances_[selected_indices]
            self.feature_scores = dict(zip(self.selected_features, variances.tolist()))

        elif self.method == 'mutual_info':
            mi_scores = mutual_info_classif(X, y, random_state=42)
            top_indices = np.argsort(mi_scores)[::-1][:n_features_to_select]
            self.selected_features = list(X.columns[top_indices])
            self.feature_scores = dict(zip(self.selected_features, mi_scores[top_indices].tolist()))

        else: # Default: Random Forest Feature Importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            importances = rf.feature_importances_
            top_indices = np.argsort(importances)[::-1][:n_features_to_select]
            self.selected_features = list(X.columns[top_indices])
            self.feature_scores = dict(zip(self.selected_features, importances[top_indices].tolist()))

        X_selected = X[self.selected_features]
        return X_selected, self.selected_features, self.feature_scores

    def transform_quantum_features(self, X: pd.DataFrame, num_qubits: int = 4) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select small feature set specifically optimized for variational quantum circuit embedding (e.g. 4 qubits).
        """
        top_k = min(num_qubits, X.shape[1])
        if self.selected_features:
            q_features = self.selected_features[:top_k]
        else:
            q_features = list(X.columns[:top_k])
        return X[q_features], q_features
