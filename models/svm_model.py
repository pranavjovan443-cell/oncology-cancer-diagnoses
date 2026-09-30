import os
import joblib
import numpy as np
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVC

from config import Config
from utils.metrics import compute_comprehensive_metrics
from utils.helpers import Timer

class SVMModelWrapper:
    """
    Support Vector Machine Classifier (SVC) wrapper with probability estimation.
    """

    def __init__(self, kernel: str = 'rbf', C: float = 1.0, probability: bool = True):
        self.model_name = "SVM"
        self.model = SVC(kernel=kernel, C=C, probability=probability, random_state=Config.RANDOM_SEED)
        self.is_fitted = False
        self.metrics: Dict[str, Any] = {}
        self.training_time: float = 0.0

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train SVM model and evaluate test metrics."""
        with Timer() as timer:
            self.model.fit(X_train, y_train)

        self.training_time = timer.interval
        self.is_fitted = True

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        self.metrics = compute_comprehensive_metrics(y_test, y_pred, y_proba)
        self.metrics['training_time'] = round(self.training_time, 4)
        return self.metrics

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return binary prediction class array and positive class probability array."""
        if not self.is_fitted:
            raise ValueError("SVM model is not fitted.")
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        return y_pred, y_proba

    def save(self, filepath: str = None):
        """Save fitted model to disk."""
        filepath = filepath or os.path.join(Config.SAVED_MODELS_FOLDER, 'svm_model.pkl')
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str = None) -> 'SVMModelWrapper':
        """Load fitted model from disk."""
        filepath = filepath or os.path.join(Config.SAVED_MODELS_FOLDER, 'svm_model.pkl')
        return joblib.load(filepath)
