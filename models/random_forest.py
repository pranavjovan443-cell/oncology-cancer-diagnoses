import os
import joblib
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier

from config import Config
from utils.metrics import compute_comprehensive_metrics
from utils.helpers import Timer

class RandomForestWrapper:
    """
    Random Forest Classifier wrapper with feature importance analysis.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        self.model_name = "Random Forest"
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=Config.RANDOM_SEED
        )
        self.is_fitted = False
        self.metrics: Dict[str, Any] = {}
        self.training_time: float = 0.0

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train Random Forest classifier and evaluate test metrics."""
        with Timer() as timer:
            self.model.fit(X_train, y_train)

        self.training_time = timer.interval
        self.is_fitted = True

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        self.metrics = compute_comprehensive_metrics(y_test, y_pred, y_proba)
        self.metrics['training_time'] = round(self.training_time, 4)
        self.metrics['feature_importances'] = self.model.feature_importances_.tolist()
        return self.metrics

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return binary predictions and probability scores."""
        if not self.is_fitted:
            raise ValueError("Random Forest model is not fitted.")
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        return y_pred, y_proba

    def save(self, filepath: str = None):
        """Save fitted model to disk."""
        filepath = filepath or os.path.join(Config.SAVED_MODELS_FOLDER, 'random_forest_model.pkl')
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str = None) -> 'RandomForestWrapper':
        """Load fitted model from disk."""
        filepath = filepath or os.path.join(Config.SAVED_MODELS_FOLDER, 'random_forest_model.pkl')
        return joblib.load(filepath)
