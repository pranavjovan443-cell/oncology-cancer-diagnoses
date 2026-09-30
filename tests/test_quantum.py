import pytest
import numpy as np
from quantum.qbm import QuantumBoltzmannMachine
from models.quantum_model import QuantumModelWrapper

def test_quantum_circuit_execution():
    qbm = QuantumBoltzmannMachine(n_qubits=4, n_layers=1, epochs=2)
    X = np.random.randn(10, 4)
    y = np.random.choice([0, 1], size=10)

    loss_history = qbm.fit(X, y)
    assert len(loss_history) == 2

    proba = qbm.predict_proba(X)
    assert len(proba) == 10
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

def test_quantum_model_wrapper():
    X_train = np.random.randn(20, 4)
    y_train = np.random.choice([0, 1], size=20)
    X_test = np.random.randn(5, 4)
    y_test = np.random.choice([0, 1], size=5)

    wrapper = QuantumModelWrapper(n_qubits=4, epochs=2)
    metrics = wrapper.train(X_train, y_train, X_test, y_test)

    assert metrics['n_qubits'] == 4
    assert 'training_time' in metrics
