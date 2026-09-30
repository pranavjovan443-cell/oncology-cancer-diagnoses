import numpy as np
import pennylane as qml

def normalize_features_for_embedding(X: np.ndarray, feature_range: tuple = (0, np.pi)) -> np.ndarray:
    """
    Normalize numerical feature vector values to target angular range (default [0, pi]) for quantum gate rotations.
    """
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    
    # Avoid divide-by-zero for constant columns
    denom = np.where(X_max - X_min == 0, 1.0, X_max - X_min)
    X_std = (X - X_min) / denom
    
    a, b = feature_range
    X_scaled = X_std * (b - a) + a
    return X_scaled

def apply_angle_embedding(features: np.ndarray, wires: list, rotation: str = 'Y'):
    """
    Apply PennyLane AngleEmbedding mapping each feature scalar to single-qubit rotation gate.
    
    Args:
        features: 1D normalized feature values (length equals len(wires))
        wires: Qubit wire indices list [0, 1, 2, ...]
        rotation: 'X', 'Y', or 'Z' rotation axis
    """
    qml.AngleEmbedding(features=features, wires=wires, rotation=rotation)
