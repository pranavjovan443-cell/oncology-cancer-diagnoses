import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    """Base Configuration Class."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-oncology-qbm-secret-key-2026')
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        f"sqlite:///{BASE_DIR / 'database.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File upload configurations
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'dataset', 'raw')
    PROCESSED_FOLDER = os.path.join(BASE_DIR, 'dataset', 'processed')
    SAMPLE_FOLDER = os.path.join(BASE_DIR, 'dataset', 'sample')
    SAVED_MODELS_FOLDER = os.path.join(BASE_DIR, 'saved_models')
    REPORTS_FOLDER = os.path.join(BASE_DIR, 'reports', 'generated')
    
    ALLOWED_EXTENSIONS = {'csv'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB limit
    
    # Quantum Computing Settings
    QUANTUM_BACKEND = os.environ.get('QUANTUM_BACKEND', 'pennylane')
    DEFAULT_NUM_QUBITS = int(os.environ.get('DEFAULT_NUM_QUBITS', 4))
    DEFAULT_QUANTUM_LAYERS = int(os.environ.get('DEFAULT_QUANTUM_LAYERS', 2))
    
    # Experiment Settings
    RANDOM_SEED = 42

class DevelopmentConfig(Config):
    """Development Environment Configuration."""
    DEBUG = True
    TESTING = False

class TestingConfig(Config):
    """Testing Environment Configuration."""
    DEBUG = False
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False

class ProductionConfig(Config):
    """Production Environment Configuration."""
    DEBUG = False
    TESTING = False

config_by_name = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
