import re
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

def sanitize_user_input(text: str) -> str:
    """Sanitize text input string to protect against script injection."""
    if not isinstance(text, str):
        return ""
    # Strip dangerous HTML tags
    cleaned = re.sub(r'<[^>]*>', '', text)
    return cleaned.strip()

def safe_filename(filename: str) -> str:
    """Generate secure sanitized filename."""
    return secure_filename(filename)

def hash_password(password: str) -> str:
    """Generate pbkdf2/sha256 password hash."""
    return generate_password_hash(password)

def verify_password(password_hash: str, password: str) -> bool:
    """Verify raw password against hash."""
    return check_password_hash(password_hash, password)
