import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.preprocessing import LabelEncoder

class CategoricalEncoder:
    """
    Handles encoding of categorical predictors and target labels.
    """

    def __init__(self):
        self.target_encoder = LabelEncoder()
        self.feature_encoders: Dict[str, LabelEncoder] = {}
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame, target_column: str = 'target') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Encode target and categorical predictors.
        """
        df_encoded = df.copy()

        # 1. Encode target if categorical or string
        if target_column in df_encoded.columns:
            y_raw = df_encoded[target_column]
            if y_raw.dtype == 'object' or str(y_raw.dtype) == 'category':
                df_encoded[target_column] = self.target_encoder.fit_transform(y_raw)
            else:
                # Ensure integer 0 / 1
                df_encoded[target_column] = y_raw.astype(int)

        # 2. Encode categorical feature columns
        feature_cols = [c for c in df_encoded.columns if c != target_column]
        cat_cols = df_encoded[feature_cols].select_dtypes(include=['object', 'category']).columns

        for col in cat_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            self.feature_encoders[col] = le

        self.is_fitted = True
        return df_encoded, {
            'encoded_target': target_column,
            'categorical_features_encoded': list(cat_cols)
        }

    def transform_target(self, y: pd.Series) -> np.ndarray:
        """Transform raw target labels using fitted encoder."""
        if hasattr(self.target_encoder, 'classes_'):
            return self.target_encoder.transform(y)
        return y.values.astype(int)

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """Convert 0/1 integer predictions back to original class names."""
        if hasattr(self.target_encoder, 'classes_'):
            return self.target_encoder.inverse_transform(y)
        return np.where(y == 1, 'Drug Resistant', 'Drug Sensitive')
