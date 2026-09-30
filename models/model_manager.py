import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from config import Config
from models.svm_model import SVMModelWrapper
from models.random_forest import RandomForestWrapper
from models.deep_learning import DeepLearningModelWrapper

class ModelManager:
    """
    Central manager for orchestrating classical and quantum model training, benchmarking,
    persistence, and comparison.
    """

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.comparison_summary: List[Dict[str, Any]] = []

    def train_all_classical(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Train SVM, Random Forest, and Deep Learning models."""
        results = {}

        # 1. Train SVM
        svm = SVMModelWrapper()
        svm_metrics = svm.train(X_train, y_train, X_test, y_test)
        svm.save()
        self.models['SVM'] = svm
        results['SVM'] = svm_metrics

        # 2. Train Random Forest
        rf = RandomForestWrapper()
        rf_metrics = rf.train(X_train, y_train, X_test, y_test)
        rf.save()
        self.models['Random Forest'] = rf
        results['Random Forest'] = rf_metrics

        # 3. Train Deep Learning
        dl = DeepLearningModelWrapper(input_dim=X_train.shape[1])
        dl_metrics = dl.train(X_train, y_train, X_test, y_test)
        dl.save()
        self.models['Deep Learning'] = dl
        results['Deep Learning'] = dl_metrics

        self._update_comparison_summary()
        return results

    def add_quantum_results(self, quantum_wrapper: Any, metrics: Dict[str, Any]):
        """Register quantum / QBM model results into manager."""
        self.models['Quantum Model'] = quantum_wrapper
        self._update_comparison_summary()

    def get_comparison_table(self) -> pd.DataFrame:
        """
        Generate comparative metrics table for all trained models.
        """
        rows = []
        for model_name, wrapper in self.models.items():
            metrics = getattr(wrapper, 'metrics', {})
            rows.append({
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0.0),
                'Precision': metrics.get('precision', 0.0),
                'Recall': metrics.get('recall', 0.0),
                'F1-Score': metrics.get('f1_score', 0.0),
                'ROC-AUC': metrics.get('roc_auc', 0.0),
                'Training Time (s)': metrics.get('training_time', 0.0)
            })
        return pd.DataFrame(rows)

    def _update_comparison_summary(self):
        """Internal helper to refresh comparison list."""
        df = self.get_comparison_table()
        self.comparison_summary = df.to_dict(orient='records')

    def load_trained_model(self, model_name: str) -> Optional[Any]:
        """Load trained model wrapper by model_name string."""
        if model_name in self.models:
            return self.models[model_name]

        if model_name == 'SVM':
            path = os.path.join(Config.SAVED_MODELS_FOLDER, 'svm_model.pkl')
            if os.path.exists(path):
                m = SVMModelWrapper.load(path)
                self.models['SVM'] = m
                return m
        elif model_name == 'Random Forest':
            path = os.path.join(Config.SAVED_MODELS_FOLDER, 'random_forest_model.pkl')
            if os.path.exists(path):
                m = RandomForestWrapper.load(path)
                self.models['Random Forest'] = m
                return m
        elif model_name == 'Deep Learning':
            path = os.path.join(Config.SAVED_MODELS_FOLDER, 'deep_learning_model.keras')
            if os.path.exists(path):
                m = DeepLearningModelWrapper.load(path)
                self.models['Deep Learning'] = m
                return m

        return None
