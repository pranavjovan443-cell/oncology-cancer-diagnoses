import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

class DataCleaner:
    """
    Cleans raw oncology dataset by handling missing values, duplicates, and invalid entries.
    """

    def __init__(self, strategy: str = 'impute'):
        """
        Args:
            strategy: 'impute' (median/mode) or 'drop' (remove incomplete rows).
        """
        self.strategy = strategy

    def fit_transform(self, df: pd.DataFrame, target_column: str = 'target') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean dataset and return cleaned DataFrame with transformation summary.
        """
        df_clean = df.copy()
        summary = {
            'initial_rows': len(df),
            'initial_cols': len(df.columns),
            'duplicates_removed': 0,
            'missing_imputed': 0,
            'rows_dropped': 0
        }

        # 1. Remove exact duplicate rows
        duplicates = df_clean.duplicated().sum()
        if duplicates > 0:
            df_clean = df_clean.drop_duplicates()
            summary['duplicates_removed'] = int(duplicates)

        # 2. Separate features and target
        features_df = df_clean.drop(columns=[target_column]) if target_column in df_clean.columns else df_clean

        # 3. Handle missing values
        if self.strategy == 'drop':
            initial_count = len(df_clean)
            df_clean = df_clean.dropna()
            summary['rows_dropped'] = initial_count - len(df_clean)
        else:
            # Impute strategy
            num_cols = features_df.select_dtypes(include=[np.number]).columns
            cat_cols = features_df.select_dtypes(exclude=[np.number]).columns

            for col in num_cols:
                if df_clean[col].isnull().sum() > 0:
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
                    summary['missing_imputed'] += 1

            for col in cat_cols:
                if df_clean[col].isnull().sum() > 0:
                    mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                    df_clean[col] = df_clean[col].fillna(mode_val)
                    summary['missing_imputed'] += 1

        summary['final_rows'] = len(df_clean)
        summary['final_cols'] = len(df_clean.columns)

        return df_clean, summary
