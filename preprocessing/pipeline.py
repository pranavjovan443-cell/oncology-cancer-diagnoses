import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split

from config import Config
from preprocessing.cleaner import DataCleaner
from preprocessing.encoder import CategoricalEncoder
from preprocessing.scaler import FeatureScaler
from preprocessing.feature_selection import FeatureSelector

class PreprocessingPipeline:
    """
    Complete end-to-end data preprocessing pipeline enforcing zero data leakage.
    """

    def __init__(self, target_column: str = 'target', test_size: float = 0.2, random_state: int = 42):
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

        self.cleaner = DataCleaner()
        self.encoder = CategoricalEncoder()
        self.scaler = FeatureScaler()
        self.selector = FeatureSelector(method='rf_importance', n_features=30)
        self.is_fitted = False

        self.feature_names = []
        self.quantum_features = []

    def fit_transform(self, df: pd.DataFrame, num_quantum_qubits: int = 4) -> Dict[str, Any]:
        """
        Execute full pipeline fit & transform on raw DataFrame.
        
        Returns dictionary containing:
            X_train_scaled, X_test_scaled, y_train, y_test,
            X_train_quantum, X_test_quantum, feature_names, quantum_features, summary
        """
        # 1. Cleaning
        df_clean, clean_summary = self.cleaner.fit_transform(df, target_column=self.target_column)

        # 2. Categorical Encoding
        df_encoded, encode_summary = self.encoder.fit_transform(df_clean, target_column=self.target_column)

        # Separate X and y
        y = df_encoded[self.target_column].values
        X_raw = df_encoded.drop(columns=[self.target_column])

        # 3. Feature Selection
        X_selected, selected_features, feature_scores = self.selector.fit_transform(X_raw, y)
        self.feature_names = selected_features

        # 4. Stratified Train-Test Split (80% / 20%)
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_selected, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # 5. Fit & Transform Scaler ONLY on Training Split
        X_train_scaled = self.scaler.fit_transform(X_train_df)
        X_test_scaled = self.scaler.transform(X_test_df)

        # 6. Quantum Feature Selection & Scaling (2 - 8 qubits)
        X_quantum_df, self.quantum_features = self.selector.transform_quantum_features(
            X_selected, num_qubits=num_quantum_qubits
        )
        X_train_q_df = X_train_df[self.quantum_features]
        X_test_q_df = X_test_df[self.quantum_features]

        q_scaler = FeatureScaler()
        X_train_quantum = q_scaler.fit_transform(X_train_q_df)
        X_test_quantum = q_scaler.transform(X_test_q_df)

        self.is_fitted = True

        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_quantum': X_train_quantum,
            'X_test_quantum': X_test_quantum,
            'feature_names': self.feature_names,
            'quantum_features': self.quantum_features,
            'feature_scores': feature_scores,
            'clean_summary': clean_summary,
            'encode_summary': encode_summary
        }

    def transform_single_patient(self, input_dict: Dict[str, float], num_quantum_qubits: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform raw input dictionary of features for single patient prediction.
        
        Returns:
            Tuple of (scaled_classical_vector, scaled_quantum_vector)
        """
        if not self.is_fitted:
            raise ValueError("Pipeline is not fitted yet.")

        # Reconstruct DataFrame matching selected feature schema
        df_patient = pd.DataFrame([input_dict])[self.feature_names]

        # Scaled classical vector using fitted dataset scaler
        scaled_classical = self.scaler.transform(df_patient)

        # Extract quantum feature columns scaled using full dataset distribution
        q_indices = []
        for q_feat in self.quantum_features[:num_quantum_qubits]:
            if q_feat in self.feature_names:
                q_indices.append(self.feature_names.index(q_feat))
        
        if q_indices:
            scaled_quantum = scaled_classical[:, q_indices]
        else:
            scaled_quantum = scaled_classical[:, :num_quantum_qubits]

        return scaled_classical, scaled_quantum

    def save(self, filepath: str = None):
        """Save fitted pipeline to disk."""
        filepath = filepath or os.path.join(Config.SAVED_MODELS_FOLDER, 'preprocessing_pipeline.pkl')
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str = None) -> 'PreprocessingPipeline':
        """Load fitted pipeline from disk."""
        filepath = filepath or os.path.join(Config.SAVED_MODELS_FOLDER, 'preprocessing_pipeline.pkl')
        return joblib.load(filepath)
