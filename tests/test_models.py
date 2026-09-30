import pytest
import numpy as np
from models.svm_model import SVMModelWrapper
from models.random_forest import RandomForestWrapper
from models.deep_learning import DeepLearningModelWrapper

@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    X_train = np.random.randn(80, 10)
    y_train = np.random.choice([0, 1], size=80)
    X_test = np.random.randn(20, 10)
    y_test = np.random.choice([0, 1], size=20)
    return X_train, y_train, X_test, y_test

def test_svm_model(synthetic_data):
    X_tr, y_tr, X_te, y_te = synthetic_data
    model = SVMModelWrapper()
    metrics = model.train(X_tr, y_tr, X_te, y_te)

    assert 'accuracy' in metrics
    assert 0.0 <= metrics['accuracy'] <= 1.0

    pred, proba = model.predict(X_te)
    assert len(pred) == 20
    assert len(proba) == 20

def test_random_forest_model(synthetic_data):
    X_tr, y_tr, X_te, y_te = synthetic_data
    model = RandomForestWrapper()
    metrics = model.train(X_tr, y_tr, X_te, y_te)

    assert metrics['f1_score'] >= 0.0
    assert 'feature_importances' in metrics

def test_deep_learning_model(synthetic_data):
    X_tr, y_tr, X_te, y_te = synthetic_data
    model = DeepLearningModelWrapper(input_dim=10)
    metrics = model.train(X_tr, y_tr, X_te, y_te, epochs=5)

    assert 'loss_history' in metrics
    assert len(metrics['loss_history']) > 0
