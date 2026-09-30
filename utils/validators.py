import re
import os
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional

def validate_email(email: str) -> bool:
    """Validate email address format."""
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return bool(re.match(pattern, email))

def validate_password_strength(password: str) -> Tuple[bool, str]:
    """Validate password strength requirements."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter."
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one number."
    return True, "Password is valid."

def validate_csv_upload(filepath: str, allowed_extensions: set, max_bytes: int) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Validate uploaded CSV dataset file.
    
    Returns:
        Tuple of (is_valid, error_or_success_message, DataFrame or None)
    """
    if not os.path.exists(filepath):
        return False, "File does not exist.", None

    ext = filepath.rsplit('.', 1)[-1].lower() if '.' in filepath else ''
    if ext not in allowed_extensions:
        return False, f"Invalid file format '.{ext}'. Only .csv files are supported.", None

    file_size = os.path.getsize(filepath)
    if file_size == 0:
        return False, "Uploaded file is empty.", None
    if file_size > max_bytes:
        return False, f"File size exceeds maximum limit of {max_bytes // (1024 * 1024)}MB.", None

    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return False, "Uploaded CSV file contains no data rows.", None
        if len(df.columns) < 2:
            return False, "Dataset must contain at least 2 columns (features + target).", None
        return True, "CSV file validated successfully.", df
    except Exception as e:
        return False, f"Malformed CSV file: {str(e)}", None

def validate_target_column(df: pd.DataFrame, target_column: str) -> Tuple[bool, str]:
    """Validate target column existence and unique class distribution."""
    if target_column not in df.columns:
        return False, f"Target column '{target_column}' not found in dataset columns."
    
    unique_values = df[target_column].dropna().unique()
    if len(unique_values) < 2:
        return False, f"Target column '{target_column}' must have at least 2 distinct binary classes for resistance classification."
    if len(unique_values) > 10:
        return False, f"Target column '{target_column}' has too many unique values ({len(unique_values)}) for classification."

    return True, "Target column is valid."

def validate_prediction_input(input_data: Dict[str, Any], expected_features: List[str]) -> Tuple[bool, str, Dict[str, float]]:
    """
    Validate single patient feature prediction payload against expected feature schema.
    Gracefully fills any unprovided feature inputs with 0.0 fallback instead of failing request.
    """
    if not expected_features:
        return False, "Expected feature schema is empty.", {}

    validated_vector = {}
    provided_count = 0

    for feature in expected_features:
        if feature in input_data and input_data[feature] is not None and str(input_data[feature]).strip() != '':
            try:
                val = float(input_data[feature])
                validated_vector[feature] = val
                provided_count += 1
            except (ValueError, TypeError):
                validated_vector[feature] = 0.0
        else:
            validated_vector[feature] = 0.0

    # If completely empty payload or zero matching features provided
    if provided_count == 0 and len(input_data) > 0 and not any(k in expected_features for k in input_data.keys()):
        missing_features = [f for f in expected_features if f not in input_data]
        return False, f"Missing required features: {', '.join(missing_features[:5])}", {}

    return True, "Input feature vector validated.", validated_vector
