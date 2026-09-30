import numpy as np
import pennylane as qml
from typing import Any
from quantum.embedding import apply_angle_embedding

def build_variational_circuit(n_qubits: int = 4, n_layers: int = 2, dev: Any = None):
    """
    Construct a PennyLane QNode executing a Variational Quantum Circuit (VQC) with CNOT entanglement.
    
    Architecture:
        1. Feature Encoding (AngleEmbedding RY)
        2. Trainable Variational Layers:
           - Parameterized RY & RZ rotations on each qubit
           - Ring / Ladder CNOT entanglement pattern
        3. Measurement: PauliZ expectation value on Qubit 0
    """
    if dev is None:
        dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(weights, features):
        # 1. Quantum Feature Encoding
        apply_angle_embedding(features, wires=range(n_qubits), rotation='Y')

        # 2. Variational Layers
        for layer in range(n_layers):
            # Parameterized Single-Qubit Rotations
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)

            # Entanglement (Ring topology)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])

        # 3. Measurement: PauliZ expectation on wire 0
        return qml.expval(qml.PauliZ(0))

    return circuit
