import pennylane as qml
from typing import Optional, Any

def get_quantum_device(backend_name: str = 'pennylane', n_qubits: int = 4) -> Any:
    """
    Instantiate and return PennyLane quantum simulation device.
    Supports local default.qubit or Qiskit Aer backend simulation without cloud credentials.
    """
    backend_clean = str(backend_name).lower().strip()

    if backend_clean in ['qiskit', 'qiskit_aer', 'qiskit.aer']:
        try:
            # Attempt to initialize Qiskit Aer simulator plugin if available
            dev = qml.device("qiskit.aer", wires=n_qubits)
            print(f"Initialized Qiskit Aer Quantum Simulator backend ({n_qubits} qubits).")
            return dev
        except Exception as e:
            print(f"Qiskit Aer plugin initialization note: {e}. Falling back to PennyLane default.qubit simulator.")
            return qml.device("default.qubit", wires=n_qubits)
    else:
        # Default PennyLane statevector simulator
        return qml.device("default.qubit", wires=n_qubits)
