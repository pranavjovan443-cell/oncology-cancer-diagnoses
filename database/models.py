from datetime import datetime, timezone
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from database import db

def utc_now():
    return datetime.now(timezone.utc)

class User(UserMixin, db.Model):
    """User account model for authentication and role management."""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=utc_now, nullable=False)

    datasets = db.relationship('UploadedDataset', backref='uploader', lazy=True)

    def set_password(self, password: str) -> None:
        """Hash and set user password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Check user password hash match."""
        return check_password_hash(self.password_hash, password)

    def __repr__(self) -> str:
        return f"<User {self.username}>"

class UploadedDataset(db.Model):
    """Uploaded dataset metadata and tracking model."""
    __tablename__ = 'uploaded_datasets'

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(512), nullable=False)
    rows = db.Column(db.Integer, nullable=False)
    columns = db.Column(db.Integer, nullable=False)
    target_column = db.Column(db.String(128), nullable=False, default='target')
    uploaded_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=utc_now, nullable=False)

    patients = db.relationship('Patient', backref='dataset', lazy=True)

    def __repr__(self) -> str:
        return f"<UploadedDataset {self.filename} ({self.rows}x{self.columns})>"

class Patient(db.Model):
    """Anonymized patient clinical record model."""
    __tablename__ = 'patients'

    id = db.Column(db.Integer, primary_key=True)
    patient_identifier = db.Column(db.String(128), unique=True, nullable=False, index=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('uploaded_datasets.id'), nullable=True)
    features = db.Column(db.Text, nullable=False)  # JSON-encoded dictionary of feature values
    created_at = db.Column(db.DateTime, default=utc_now, nullable=False)

    predictions = db.relationship('Prediction', backref='patient', lazy=True)

    def __repr__(self) -> str:
        return f"<Patient {self.patient_identifier}>"

class Prediction(db.Model):
    """Model inference prediction record."""
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    model_name = db.Column(db.String(64), nullable=False)
    prediction = db.Column(db.String(64), nullable=False)  # e.g., 'Drug Resistant' or 'Drug Sensitive'
    probability = db.Column(db.Float, nullable=False)       # Output probability (0.0 to 1.0)
    details_json = db.Column(db.Text, nullable=True)       # Additional details in JSON
    created_at = db.Column(db.DateTime, default=utc_now, nullable=False)

    def __repr__(self) -> str:
        return f"<Prediction {self.model_name} -> {self.prediction} ({self.probability:.2f})>"

class ExperimentResult(db.Model):
    """Machine learning and Quantum model benchmark result model."""
    __tablename__ = 'experiment_results'

    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(64), nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    precision = db.Column(db.Float, nullable=False)
    recall = db.Column(db.Float, nullable=False)
    f1_score = db.Column(db.Float, nullable=False)
    roc_auc = db.Column(db.Float, nullable=False)
    training_time = db.Column(db.Float, nullable=False)  # Time in seconds
    metrics_json = db.Column(db.Text, nullable=True)     # Detailed metrics dict (confusion matrix, ROC, etc.)
    created_at = db.Column(db.DateTime, default=utc_now, nullable=False)

    def __repr__(self) -> str:
        return f"<ExperimentResult {self.model_name} Acc={self.accuracy:.4f}>"
