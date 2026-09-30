import os
import sys
from pathlib import Path

# Add project root directory to sys.path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.datasets import load_breast_cancer

from config import Config
from database import db
from database.models import User, UploadedDataset, Patient, Prediction, ExperimentResult

def ensure_directories():
    """Ensure essential data, model, and report directories exist."""
    directories = [
        Config.UPLOAD_FOLDER,
        Config.PROCESSED_FOLDER,
        Config.SAMPLE_FOLDER,
        Config.SAVED_MODELS_FOLDER,
        Config.REPORTS_FOLDER
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        gitkeep = os.path.join(directory, '.gitkeep')
        if not os.path.exists(gitkeep):
            with open(gitkeep, 'w') as f:
                f.write('')

def create_sample_dataset() -> str:
    """
    Creates a demonstration breast cancer dataset from scikit-learn.
    Saves to dataset/sample/breast_cancer_demo.csv.
    """
    ensure_directories()
    sample_path = os.path.join(Config.SAMPLE_FOLDER, 'breast_cancer_demo.csv')
    if not os.path.exists(sample_path):
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        # Rename features to valid python identifier style column names
        df.columns = [c.replace(' ', '_') for c in df.columns]
        # Target: 1 = Malignant (Drug Resistant proxy), 0 = Benign (Drug Sensitive proxy)
        # In sklearn, target 0 = malignant, 1 = benign. Let's map 0 -> Drug Resistant (1), 1 -> Drug Sensitive (0)
        # To make it intuitive: 1 = Resistant, 0 = Sensitive
        df['target'] = (data.target == 0).astype(int)
        df.to_csv(sample_path, index=False)
        print(f"Sample dataset generated at {sample_path}")
    return sample_path

def init_db(app):
    """Initialize the SQLite database schema and seed essential demo data."""
    ensure_directories()
    sample_csv_path = create_sample_dataset()

    with app.app_context():
        db.create_all()

        # Seed default admin researcher user if no users exist
        if User.query.filter_by(username='researcher').first() is None and User.query.filter_by(email='researcher@oncology.lab').first() is None:
            default_user = User(username='researcher', email='researcher@oncology.lab')
            default_user.set_password('ResearchPass2026!')
            db.session.add(default_user)
            db.session.commit()
            print("Default user 'researcher' created with password 'ResearchPass2026!'.")

        # Seed sample dataset metadata if not already recorded
        if UploadedDataset.query.filter_by(filename='breast_cancer_demo.csv').first() is None:
            df = pd.read_csv(sample_csv_path)
            demo_dataset = UploadedDataset(
                filename='breast_cancer_demo.csv',
                filepath=sample_csv_path,
                rows=len(df),
                columns=len(df.columns),
                target_column='target',
                uploaded_by=1
            )
            db.session.add(demo_dataset)
            db.session.commit()
            print("Demo dataset registered in SQLite database.")

if __name__ == '__main__':
    from flask import Flask
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)
    init_db(app)
    print("Database initialization completed successfully.")
