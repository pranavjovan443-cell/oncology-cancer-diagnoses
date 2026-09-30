import pytest
import numpy as np
import pandas as pd
from app import app, db
from preprocessing.pipeline import PreprocessingPipeline
from models.random_forest import RandomForestWrapper

@pytest.fixture
def client():
    orig_uri = app.config.get('SQLALCHEMY_DATABASE_URI')
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['WTF_CSRF_ENABLED'] = False

    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            yield client
            db.drop_all()

    if orig_uri:
        app.config['SQLALCHEMY_DATABASE_URI'] = orig_uri

def test_prediction_pipeline(client):
    # Train pipeline and model on realistic dataset schema
    df = pd.read_csv('dataset/sample/breast_cancer_demo.csv')

    pipeline = PreprocessingPipeline(target_column='target')
    data = pipeline.fit_transform(df)
    pipeline.save()

    rf = RandomForestWrapper()
    rf.train(data['X_train'], data['y_train'], data['X_test'], data['y_test'])
    rf.save()

    # Login researcher user
    client.post('/register', data={
        'username': 'tester',
        'email': 'tester@lab.org',
        'password': 'Password123!',
        'confirm_password': 'Password123!'
    })
    client.post('/login', data={'username': 'tester', 'password': 'Password123!'})

    features = {f: 0.5 for f in pipeline.feature_names}
    response = client.post('/api/predict', json={
        'patient_id': 'PAT-TEST01',
        'model_name': 'Quantum Boltzmann Machine (QBM)',
        'features': features
    })

    assert response.status_code == 200
    res_data = response.get_json()
    assert res_data['patient_id'] == 'PAT-TEST01'
    assert res_data['prediction'] in ['Drug Resistant', 'Drug Sensitive']
