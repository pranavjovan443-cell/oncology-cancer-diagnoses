import os
import joblib
import numpy as np
from typing import Dict, Any, Tuple

from config import Config
from quantum.qbm import QuantumBoltzmannMachine
from utils.metrics import compute_comprehensive_metrics
from utils.helpers import Timer

class QuantumModelWrapper:
    """
    Scikit-learn compatible wrapper for Quantum / QBM-Inspired Variational Classifier.
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 2, backend: str = None, lr: float = 0.05, epochs: int = 20):
        self.model_name = "Quantum Model"
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend = backend or Config.QUANTUM_BACKEND
        self.lr = lr
        self.epochs = epochs

        self.qbm = QuantumBoltzmannMachine(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            backend=self.backend,
            lr=self.lr,
            epochs=self.epochs
        )

        self.is_fitted = False
        self.metrics: Dict[str, Any] = {}
        self.training_time: float = 0.0
        self.loss_history = []

    def train(self, X_train_q: np.ndarray, y_train: np.ndarray, X_test_q: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train variational quantum circuit classifier on dimensionally reduced quantum features."""
        with Timer() as timer:
            self.loss_history = self.qbm.fit(X_train_q, y_train)

        self.training_time = timer.interval
        self.is_fitted = True

        y_proba = self.qbm.predict_proba(X_test_q)
        y_pred = (y_proba >= 0.5).astype(int)

        self.metrics = compute_comprehensive_metrics(y_test, y_pred, y_proba)
        self.metrics['training_time'] = round(self.training_time, 4)
        self.metrics['loss_history'] = self.loss_history
        self.metrics['n_qubits'] = self.n_qubits
        self.metrics['n_layers'] = self.n_layers
        self.metrics['backend'] = self.backend
        return self.metrics

    def predict(self, X_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return binary prediction labels and probability outputs."""
        if not self.is_fitted:
            raise ValueError("Quantum model is not trained.")
        y_proba = self.qbm.predict_proba(X_q)
        y_pred = (y_proba >= 0.5).astype(int)
        return y_pred, y_proba

    def save(self, filepath: str = None):
        """Save quantum model wrapper and parameters to disk."""
        filepath = filepath or os.path.join(Config.SAVED_MODELS_FOLDER, 'quantum_model.pkl')
        meta = {
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'backend': self.backend,
            'weights': self.qbm.weights,
            'bias': self.qbm.bias,
            'metrics': self.metrics,
            'training_time': self.training_time,
            'loss_history': self.loss_history,
            'is_fitted': self.is_fitted
        }
        joblib.dump(meta, filepath)

    @staticmethod
    def load(filepath: str = None) -> 'QuantumModelWrapper':
        """Load trained quantum model from disk."""
        filepath = filepath or os.path.join(Config.SAVED_MODELS_FOLDER, 'quantum_model.pkl')
        meta = joblib.load(filepath)
        
        wrapper = QuantumModelWrapper(
            n_qubits=meta['n_qubits'],
            n_layers=meta['n_layers'],
            backend=meta['backend']
        )
        wrapper.qbm.weights = meta['weights']
        wrapper.qbm.bias = meta['bias']
        wrapper.qbm.is_fitted = True
        wrapper.metrics = meta['metrics']
        wrapper.training_time = meta['training_time']
        wrapper.loss_history = meta['loss_history']
        wrapper.is_fitted = meta['is_fitted']

        return wrapper
