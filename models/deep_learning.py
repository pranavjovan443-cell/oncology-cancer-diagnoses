import os
import joblib
import numpy as np
from typing import Dict, Any, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from config import Config
from utils.metrics import compute_comprehensive_metrics
from utils.helpers import Timer

# Set random seed for reproducibility
tf.random.set_seed(Config.RANDOM_SEED)

class DeepLearningModelWrapper:
    """
    Lightweight TensorFlow/Keras Neural Network for binary drug resistance classification.
    """

    def __init__(self, input_dim: int = 15, hidden_units: int = 32, dropout_rate: float = 0.2):
        self.model_name = "Deep Learning"
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate

        self.model = self._build_model()
        self.is_fitted = False
        self.metrics: Dict[str, Any] = {}
        self.loss_history = []
        self.val_loss_history = []
        self.training_time: float = 0.0

    def _build_model(self) -> Sequential:
        """Construct sequential multi-layer perceptron neural network."""
        model = Sequential([
            Input(shape=(self.input_dim,)),
            Dense(self.hidden_units, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(self.hidden_units // 2, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, epochs: int = 40, batch_size: int = 16) -> Dict[str, Any]:
        """Train Deep Learning model with early stopping on validation split."""
        if X_train.shape[1] != self.input_dim:
            self.input_dim = X_train.shape[1]
            self.model = self._build_model()

        early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

        with Timer() as timer:
            history = self.model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=0
            )

        self.training_time = timer.interval
        self.is_fitted = True

        self.loss_history = [float(x) for x in history.history['loss']]
        self.val_loss_history = [float(x) for x in history.history.get('val_loss', [])]

        y_proba = self.model.predict(X_test, verbose=0).ravel()
        y_pred = (y_proba >= 0.5).astype(int)

        self.metrics = compute_comprehensive_metrics(y_test, y_pred, y_proba)
        self.metrics['training_time'] = round(self.training_time, 4)
        self.metrics['loss_history'] = self.loss_history
        self.metrics['val_loss_history'] = self.val_loss_history
        return self.metrics

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return binary prediction classes and output probabilities."""
        if not self.is_fitted:
            raise ValueError("Deep Learning model is not trained.")
        y_proba = self.model.predict(X, verbose=0).ravel()
        y_pred = (y_proba >= 0.5).astype(int)
        return y_pred, y_proba

    def save(self, filepath: str = None):
        """Save model architecture and weights to disk."""
        filepath = filepath or os.path.join(Config.SAVED_MODELS_FOLDER, 'deep_learning_model.keras')
        self.model.save(filepath)

        # Save wrapper metadata
        meta_path = filepath + '.wrapper.pkl'
        meta = {
            'input_dim': self.input_dim,
            'metrics': self.metrics,
            'training_time': self.training_time,
            'loss_history': self.loss_history,
            'val_loss_history': self.val_loss_history,
            'is_fitted': self.is_fitted
        }
        joblib.dump(meta, meta_path)

    @staticmethod
    def load(filepath: str = None) -> 'DeepLearningModelWrapper':
        """Load trained neural network model from disk."""
        filepath = filepath or os.path.join(Config.SAVED_MODELS_FOLDER, 'deep_learning_model.keras')
        wrapper = DeepLearningModelWrapper()
        wrapper.model = load_model(filepath)

        meta_path = filepath + '.wrapper.pkl'
        if os.path.exists(meta_path):
            meta = joblib.load(meta_path)
            wrapper.input_dim = meta.get('input_dim', 15)
            wrapper.metrics = meta.get('metrics', {})
            wrapper.training_time = meta.get('training_time', 0.0)
            wrapper.loss_history = meta.get('loss_history', [])
            wrapper.val_loss_history = meta.get('val_loss_history', [])
            wrapper.is_fitted = meta.get('is_fitted', True)

        return wrapper
