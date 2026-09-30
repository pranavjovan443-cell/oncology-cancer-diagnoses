import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

class FeatureScaler:
    """
    StandardScaler wrapper ensuring zero data leakage between train and test splits.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names: List[str] = []

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit scaler on training data and return scaled array."""
        self.feature_names = list(X_train.columns)
        X_scaled = self.scaler.fit_transform(X_train)
        self.is_fitted = True
        return X_scaled

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform test data or single patient input vector using fitted parameters."""
        if not self.is_fitted:
            raise ValueError("FeatureScaler is not fitted yet. Call fit_transform first.")
        if isinstance(X, pd.DataFrame):
            # Ensure column order matches fitting order
            X = X[self.feature_names]
        return self.scaler.transform(X)
