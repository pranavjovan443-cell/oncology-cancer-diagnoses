import pytest
import pandas as pd
import numpy as np
from preprocessing.cleaner import DataCleaner
from preprocessing.encoder import CategoricalEncoder
from preprocessing.scaler import FeatureScaler
from preprocessing.feature_selection import FeatureSelector
from preprocessing.pipeline import PreprocessingPipeline

def test_data_cleaner():
    df = pd.DataFrame({
        'feature_1': [1.0, 2.0, np.nan, 4.0, 1.0],
        'feature_2': ['A', 'B', 'A', np.nan, 'A'],
        'target': [0, 1, 0, 1, 0]
    })
    cleaner = DataCleaner(strategy='impute')
    df_clean, summary = cleaner.fit_transform(df, target_column='target')

    assert df_clean['feature_1'].isnull().sum() == 0
    assert df_clean['feature_2'].isnull().sum() == 0
    assert summary['duplicates_removed'] == 1

def test_preprocessing_pipeline():
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(50, 10), columns=[f"gene_{i}" for i in range(10)])
    df['target'] = np.random.choice([0, 1], size=50)

    pipeline = PreprocessingPipeline(target_column='target')
    results = pipeline.fit_transform(df, num_quantum_qubits=4)

    assert results['X_train'].shape[0] == 40
    assert results['X_test'].shape[0] == 10
    assert results['X_train_quantum'].shape[1] == 4
    assert len(results['feature_names']) > 0
