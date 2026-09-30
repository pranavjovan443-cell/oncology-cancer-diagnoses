import time
import uuid
import numpy as np
from typing import Any, Dict

def generate_patient_id() -> str:
    """Generate unique anonymized patient identifier (e.g. PAT-8F3A29)."""
    return f"PAT-{uuid.uuid4().hex[:6].upper()}"

def format_percentage(val: float, decimals: int = 1) -> str:
    """Format float probability/ratio as percentage string (e.g., 0.824 -> 82.4%)."""
    return f"{val * 100:.{decimals}f}%"

def make_json_serializable(obj: Any) -> Any:
    """Recursively convert NumPy objects into standard Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    return obj

class Timer:
    """Context manager for accurate runtime execution measurement in seconds."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
