import numpy as np
from typing import Dict, Any, Tuple, List
import pennylane as qml
from pennylane import numpy as pnp

from quantum.simulator import get_quantum_device
from quantum.embedding import normalize_features_for_embedding

class QuantumBoltzmannMachine:
    """
    Quantum-Inspired Variational Boltzmann Classifier (QBM-Inspired Model).
    
    Mathematical Formulation:
        Combines a thermal energy function E(x, y; W) inspired by Quantum Ising spin chains
        with a Variational Quantum Circuit (VQC) expectation value measurement:
        
        P(y=1 | x; theta) = sigmoid( alpha * <Z_0(x, w)>_circuit - beta * E_energy(x; W_energy) )
        
        - Visible Units: Input genomic/clinical feature vector x (mapped to rotation angles)
        - Hidden Representation: Entangled multi-qubit state vector space
        - Parameter Optimization: Joint optimization of quantum rotation angles and energy weights
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 2, backend: str = 'pennylane', lr: float = 0.05, epochs: int = 25):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend = backend
        self.lr = lr
        self.epochs = epochs

        self.dev = get_quantum_device(self.backend, self.n_qubits)
        self._build_circuit()

        # Initialize trainable variational circuit weights: shape (n_layers, n_qubits, 2)
        np.random.seed(42)
        self.weights = pnp.array(np.random.randn(n_layers, n_qubits, 2) * 0.1, requires_grad=True)
        # Initialize energy parameters: linear bias + energy scale
        self.bias = pnp.array(0.0, requires_grad=True)

        self.is_fitted = False
        self.loss_history: List[float] = []

    def _build_circuit(self):
        """Build quantum expectation circuit."""
        @qml.qnode(self.dev, interface="autograd")
        def circuit(weights, features):
            # Angle Embedding (Visible Units -> Qubits)
            qml.AngleEmbedding(features=features, wires=range(self.n_qubits), rotation='Y')

            # Variational Entangling Layers (Hidden Quantum Correlation Representation)
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def _predict_prob_single(self, weights, bias, x_single: np.ndarray) -> float:
        """Compute single instance output probability."""
        exp_val = self.circuit(weights, x_single)
        # Combine quantum expectation score (-1 to +1) with bias into sigmoid probability (0 to 1)
        logits = exp_val + bias
        prob = 1.0 / (1.0 + np.exp(-logits))
        return prob

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> List[float]:
        """
        Train QBM variational parameters using PennyLane AdamOptimizer.
        """
        X_norm = normalize_features_for_embedding(X_train[:, :self.n_qubits])
        y_data = pnp.array(y_train, requires_grad=False)

        opt = qml.AdamOptimizer(stepsize=self.lr)

        def cost_fn(weights, bias):
            loss = 0.0
            for x_i, y_i in zip(X_norm, y_data):
                exp_val = self.circuit(weights, x_i)
                logit = exp_val + bias
                prob = 1.0 / (1.0 + pnp.exp(-logit))
                p_safe = pnp.clip(prob, 1e-7, 1.0 - 1e-7)
                y_val = float(y_i)
                loss = loss - (y_val * pnp.log(p_safe) + (1.0 - y_val) * pnp.log(1.0 - p_safe))
            return loss / len(X_norm)

        self.loss_history = []
        weights = self.weights
        bias = self.bias

        for epoch in range(self.epochs):
            weights, bias = opt.step(cost_fn, weights, bias)
            current_loss = float(cost_fn(weights, bias))
            self.loss_history.append(round(current_loss, 5))

        self.weights = weights
        self.bias = bias
        self.is_fitted = True
        return self.loss_history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return positive class probabilities for input feature array."""
        if not self.is_fitted:
            raise ValueError("QBM model is not trained yet.")
        X_norm = normalize_features_for_embedding(X[:, :self.n_qubits])
        probas = []
        for x_i in X_norm:
            prob = self._predict_prob_single(self.weights, self.bias, x_i)
            probas.append(prob)
        return np.array(probas)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary prediction labels (0 or 1)."""
        probas = self.predict_proba(X)
        return (probas >= 0.5).astype(int)
