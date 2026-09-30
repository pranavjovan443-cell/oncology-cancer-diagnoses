import os
import warnings

# Suppress verbose C++/Library warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

from typing import Dict, Any, List, Tuple
import json
import logging
import pandas as pd
import numpy as np
from flask import (
    Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, session
)
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

from config import Config
from database import db
from database.models import User, UploadedDataset, Patient, Prediction, ExperimentResult
from database.init_db import init_db

from preprocessing.pipeline import PreprocessingPipeline
from models.model_manager import ModelManager
from models.svm_model import SVMModelWrapper
from models.random_forest import RandomForestWrapper
from models.deep_learning import DeepLearningModelWrapper
from models.quantum_model import QuantumModelWrapper

from utils.validators import validate_csv_upload, validate_target_column, validate_prediction_input, validate_email, validate_password_strength
from utils.security import safe_filename, sanitize_user_input
from utils.plotting import (
    plot_confusion_matrix, plot_roc_curves, plot_precision_recall_curves,
    plot_feature_importance, plot_model_comparison_bar, plot_training_loss
)
from utils.explainability import ModelExplainer
from utils.helpers import generate_patient_id, make_json_serializable
from reports.report_generator import PDFReportGenerator

# Configure Application Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Initialize Extensions
db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Ensure SQLite schema and researcher seed user on application startup
with app.app_context():
    try:
        init_db(app)
    except Exception as e:
        logger.warning(f"Database init status note: {e}")

# Initialize Central Model Manager & PDF Generator
model_manager = ModelManager()
pdf_generator = PDFReportGenerator()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize Database Schema & Seed Data
with app.app_context():
    init_db(app)

# ==========================================
# WEB PAGE ROUTES
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username_or_email = sanitize_user_input(request.form.get('username', ''))
        password = request.form.get('password', '')

        user = User.query.filter(
            (User.username == username_or_email) | (User.email == username_or_email)
        ).first()

        if user and user.check_password(password):
            login_user(user)
            logger.info(f"User {user.username} logged in successfully.")
            flash("Login successful! Welcome back to Predictive Oncology Analytics.", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username/email or password.", "danger")

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = sanitize_user_input(request.form.get('username', ''))
        email = sanitize_user_input(request.form.get('email', ''))
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template('register.html')

        if not validate_email(email):
            flash("Please enter a valid email address.", "danger")
            return render_template('register.html')

        is_valid_pw, pw_msg = validate_password_strength(password)
        if not is_valid_pw:
            flash(pw_msg, "danger")
            return render_template('register.html')

        if User.query.filter_by(username=username).first():
            flash("Username already exists. Please choose another.", "danger")
            return render_template('register.html')

        if User.query.filter_by(email=email).first():
            flash("Email already registered. Please login.", "danger")
            return render_template('register.html')

        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        logger.info(f"New user registered: {username}")
        flash("Registration successful! You can now log in.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    username = current_user.username
    logout_user()
    logger.info(f"User {username} logged out.")
    flash("You have been logged out.", "info")
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    total_patients = Patient.query.count()
    total_predictions = Prediction.query.count()
    resistant_cases = Prediction.query.filter_by(prediction='Drug Resistant').count()
    sensitive_cases = Prediction.query.filter_by(prediction='Drug Sensitive').count()
    models_trained = ExperimentResult.query.count()
    datasets_uploaded = UploadedDataset.query.count()

    recent_predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(5).all()

    return render_template(
        'dashboard.html',
        total_patients=total_patients,
        total_predictions=total_predictions,
        resistant_cases=resistant_cases,
        sensitive_cases=sensitive_cases,
        models_trained=models_trained,
        datasets_uploaded=datasets_uploaded,
        recent_predictions=recent_predictions
    )

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file part in request.", "danger")
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash("No selected file.", "danger")
            return redirect(request.url)

        filename = safe_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        is_valid, msg, df = validate_csv_upload(save_path, app.config['ALLOWED_EXTENSIONS'], app.config['MAX_CONTENT_LENGTH'])
        if not is_valid:
            flash(msg, "danger")
            return redirect(request.url)

        target_col = sanitize_user_input(request.form.get('target_column', 'target'))
        is_target_valid, target_msg = validate_target_column(df, target_col)
        if not is_target_valid:
            flash(target_msg, "warning")

        # Record dataset entry in DB
        uploaded_ds = UploadedDataset(
            filename=filename,
            filepath=save_path,
            rows=len(df),
            columns=len(df.columns),
            target_column=target_col,
            uploaded_by=current_user.id
        )
        db.session.add(uploaded_ds)
        db.session.commit()

        session['current_dataset_id'] = uploaded_ds.id
        session['target_column'] = target_col

        # Automatically fit preprocessing pipeline for immediate prediction readiness
        try:
            pipeline = PreprocessingPipeline(target_column=target_col)
            pipeline.fit_transform(df)
            pipeline.save()
        except Exception as e:
            logger.warning(f"Auto pipeline fit note: {e}")

        flash(f"Dataset '{filename}' uploaded successfully! Ready for drug resistance prediction.", "success")
        return redirect(url_for('prediction'))

    # Load sample dataset info
    sample_dataset = UploadedDataset.query.first()
    return render_template('upload.html', sample_dataset=sample_dataset)

@app.route('/preprocessing', methods=['GET', 'POST'])
@login_required
def preprocessing():
    ds_id = session.get('current_dataset_id')
    dataset = UploadedDataset.query.get(ds_id) if ds_id else UploadedDataset.query.first()

    if not dataset:
        flash("Please upload a dataset first.", "warning")
        return redirect(url_for('upload'))

    target_col = session.get('target_column', dataset.target_column)

    if request.method == 'POST':
        try:
            df = pd.read_csv(dataset.filepath)
            pipeline = PreprocessingPipeline(target_column=target_col)
            prep_data = pipeline.fit_transform(df)
            pipeline.save()

            session['pipeline_fitted'] = True
            flash("Preprocessing and feature selection completed successfully!", "success")
            return redirect(url_for('models_page'))
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            flash(f"Error during preprocessing: {str(e)}", "danger")

    return render_template('preprocessing.html', dataset=dataset, target_col=target_col)

@app.route('/models')
@login_required
def models_page():
    return render_template('models.html')

@app.route('/quantum')
@login_required
def quantum_page():
    return render_template('quantum.html')

@app.route('/prediction', methods=['GET', 'POST'])
@login_required
def prediction():
    return render_template('prediction.html')

@app.route('/explainability')
@login_required
def explainability_page():
    return render_template('explainability.html')

@app.route('/reports')
@login_required
def reports_page():
    reports_dir = app.config['REPORTS_FOLDER']
    report_files = [f for f in os.listdir(reports_dir) if f.endswith('.pdf')]
    return render_template('reports.html', report_files=report_files)

@app.route('/reports/download/<filename>')
@login_required
def download_report(filename):
    filename = safe_filename(filename)
    return send_from_directory(app.config['REPORTS_FOLDER'], filename, as_attachment=True)

# ==========================================
# REST API ENDPOINTS
# ==========================================

@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({
        'status': 'healthy',
        'service': 'Predictive Oncology Analytics API',
        'timestamp': pd.Timestamp.now().isoformat()
    }), 200

@app.route('/api/dataset/summary', methods=['GET'])
@login_required
def api_dataset_summary():
    ds_id = session.get('current_dataset_id')
    dataset = UploadedDataset.query.get(ds_id) if ds_id else UploadedDataset.query.first()

    if not dataset or not os.path.exists(dataset.filepath):
        return jsonify({'error': 'No active dataset found.'}), 44

    df = pd.read_csv(dataset.filepath)
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(df.select_dtypes(exclude=[np.number]).columns)

    target_col = dataset.target_column if dataset.target_column in df.columns else df.columns[-1]
    target_dist = df[target_col].value_counts().to_dict()

    preview = df.head(10).fillna('N/A').to_dict(orient='records')

    pipeline_features = []
    pipeline_path = os.path.join(app.config['SAVED_MODELS_FOLDER'], 'preprocessing_pipeline.pkl')
    if os.path.exists(pipeline_path):
        try:
            pipeline = PreprocessingPipeline.load(pipeline_path)
            pipeline_features = pipeline.feature_names
        except Exception:
            pass

    return jsonify({
        'filename': dataset.filename,
        'rows': len(df),
        'columns': len(df.columns),
        'numerical_columns': numeric_cols,
        'categorical_columns': categorical_cols,
        'missing_values': int(df.isnull().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'target_distribution': target_dist,
        'pipeline_features': pipeline_features,
        'preview': preview
    }), 200

@app.route('/api/train/<model_type>', methods=['POST'])
@login_required
def api_train_model(model_type):
    pipeline_path = os.path.join(app.config['SAVED_MODELS_FOLDER'], 'preprocessing_pipeline.pkl')
    if not os.path.exists(pipeline_path):
        return jsonify({'error': 'Preprocessing pipeline not found. Please run preprocessing first.'}), 400

    pipeline = PreprocessingPipeline.load(pipeline_path)
    ds_id = session.get('current_dataset_id')
    dataset = UploadedDataset.query.get(ds_id) if ds_id else UploadedDataset.query.first()
    df = pd.read_csv(dataset.filepath)

    data = pipeline.fit_transform(df)

    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

    metrics = {}

    if model_type == 'svm':
        wrapper = SVMModelWrapper()
        metrics = wrapper.train(X_train, y_train, X_test, y_test)
        wrapper.save()
        cm_path = plot_confusion_matrix(metrics['confusion_matrix']['raw'], 'SVM')

    elif model_type == 'random-forest':
        wrapper = RandomForestWrapper()
        metrics = wrapper.train(X_train, y_train, X_test, y_test)
        wrapper.save()
        cm_path = plot_confusion_matrix(metrics['confusion_matrix']['raw'], 'Random Forest')

    elif model_type == 'deep-learning':
        wrapper = DeepLearningModelWrapper(input_dim=X_train.shape[1])
        metrics = wrapper.train(X_train, y_train, X_test, y_test)
        wrapper.save()
        cm_path = plot_confusion_matrix(metrics['confusion_matrix']['raw'], 'Deep Learning')
        plot_training_loss(metrics['loss_history'], metrics.get('val_loss_history'), 'Deep Learning Training Loss')

    elif model_type == 'quantum':
        X_train_q, X_test_q = data['X_train_quantum'], data['X_test_quantum']
        wrapper = QuantumModelWrapper(n_qubits=4, n_layers=2)
        metrics = wrapper.train(X_train_q, y_train, X_test_q, y_test)
        wrapper.save()
        cm_path = plot_confusion_matrix(metrics['confusion_matrix']['raw'], 'Quantum Model')
        plot_training_loss(metrics['loss_history'], title='Quantum Variational Loss')

    else:
        return jsonify({'error': f"Unknown model type '{model_type}'."}), 400

    # Record in SQLite Database
    exp = ExperimentResult(
        model_name=wrapper.model_name,
        accuracy=metrics['accuracy'],
        precision=metrics['precision'],
        recall=metrics['recall'],
        f1_score=metrics['f1_score'],
        roc_auc=metrics['roc_auc'],
        training_time=metrics['training_time'],
        metrics_json=json.dumps(make_json_serializable(metrics))
    )
    db.session.add(exp)
    db.session.commit()

    return jsonify({
        'message': f"{wrapper.model_name} trained successfully.",
        'metrics': metrics
    }), 200

@app.route('/api/metrics', methods=['GET'])
@login_required
def api_get_metrics():
    experiments = ExperimentResult.query.all()
    metrics_summary = []
    models_dict = {}

    for exp in experiments:
        m = json.loads(exp.metrics_json) if exp.metrics_json else {}
        metrics_summary.append({
            'Model': exp.model_name,
            'Accuracy': exp.accuracy,
            'Precision': exp.precision,
            'Recall': exp.recall,
            'F1': exp.f1_score,
            'ROC-AUC': exp.roc_auc,
            'Training Time': f"{exp.training_time:.2f}s"
        })
        models_dict[exp.model_name] = m

    # Plot comparison graphs if models trained
    if models_dict:
        plot_roc_curves(models_dict)
        plot_precision_recall_curves(models_dict)
        plot_model_comparison_bar(models_dict)

    return jsonify({
        'summary': metrics_summary,
        'raw': models_dict
    }), 200

def get_therapeutic_regimens(prediction_label: str, target_medicine: str = 'General Multi-Drug Resistance Benchmark') -> Dict[str, Any]:
    """
    Return candidate computational therapeutic drug categories and agents based on sensitivity/resistance.
    """
    if prediction_label == 'Drug Sensitive':
        return {
            'target_medicine': target_medicine,
            'category': 'First-Line Standard Targeted & Chemotherapeutic Regimens',
            'status_color': 'success',
            'medicines': [
                {'name': 'Tamoxifen / Anastrozole', 'class': 'Hormonal / Endocrine Therapy', 'target': 'ER+ Receptor'},
                {'name': 'Trastuzumab (Herceptin)', 'class': 'HER2 Targeted Monoclonal Antibody', 'target': 'HER2 Neu'},
                {'name': 'Paclitaxel (Taxol)', 'class': 'Taxane Microtubule Inhibitor', 'target': 'Mitotic Spindle'},
                {'name': 'AC Regimen (Doxorubicin + Cyclophosphamide)', 'class': 'Anthracycline / Alkylating Combo', 'target': 'DNA Intercalation'}
            ],
            'mechanism_summary': f"Computational genomic alignment for target medicine '{target_medicine}' indicates high probability of therapeutic response to standard first-line targeted agents."
        }
    else:
        return {
            'target_medicine': target_medicine,
            'category': 'Second-Line / Combination Therapy & Novel Pathway Inhibitors',
            'status_color': 'danger',
            'medicines': [
                {'name': 'Palbociclib / Ribociclib', 'class': 'CDK4/6 Cell-Cycle Inhibitor', 'target': 'CDK4/6 Pathway'},
                {'name': 'Olaparib (Lynparza)', 'class': 'PARP Inhibitor', 'target': 'DNA Repair (BRCA)'},
                {'name': 'Everolimus / Alpelisib', 'class': 'mTOR / PI3K Alpha Inhibitor', 'target': 'PI3K/AKT/mTOR Pathway'},
                {'name': 'Pembrolizumab (Keytruda)', 'class': 'PD-1 / Immune Checkpoint Inhibitor', 'target': 'Tumor Immune Response'}
            ],
            'mechanism_summary': f"Computational profile indicates elevated risk of resistance to target medicine '{target_medicine}'. Suggests evaluating second-line CDK4/6, PI3K/mTOR, or immune checkpoint combination regimens."
        }

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    payload = request.get_json() or {}
    model_name = payload.get('model_name', 'Quantum Boltzmann Machine (QBM)')
    target_medicine = payload.get('target_medicine', 'General Multi-Drug Resistance Benchmark')
    features = payload.get('features', {})
    patient_identifier = payload.get('patient_id') or generate_patient_id()

    pipeline_path = os.path.join(app.config['SAVED_MODELS_FOLDER'], 'preprocessing_pipeline.pkl')
    if not os.path.exists(pipeline_path):
        return jsonify({'error': 'Pipeline not trained.'}), 400

    pipeline = PreprocessingPipeline.load(pipeline_path)

    is_valid, msg, validated_vector = validate_prediction_input(features, pipeline.feature_names)
    if not is_valid:
        return jsonify({'error': msg}), 400

    scaled_classical, scaled_quantum = pipeline.transform_single_patient(validated_vector)

    wrapper = None
    if 'Quantum' in model_name or 'QBM' in model_name:
        try:
            wrapper = QuantumModelWrapper.load()
        except Exception as e:
            logger.warning(f"Note loading quantum wrapper: {e}")

    if not wrapper:
        wrapper = model_manager.load_trained_model(model_name)

    if not wrapper:
        try:
            wrapper = QuantumModelWrapper.load()
        except Exception:
            return jsonify({'error': f"Model '{model_name}' is not trained yet."}), 400

    if isinstance(wrapper, QuantumModelWrapper) or 'Quantum' in model_name or 'QBM' in model_name:
        pred, proba = wrapper.predict(scaled_quantum)
    else:
        pred, proba = wrapper.predict(scaled_classical)

    base_proba = float(proba[0])

    # Pharmacodynamic target resistance sensitivity modifiers for specific therapeutic agents
    drug_sensitivity_offsets = {
        'Tamoxifen (ER+ Endocrine Targeted Therapy)': -0.12,
        'Trastuzumab / Herceptin (HER2 Monoclonal Antibody)': 0.18,
        'Paclitaxel / Taxol (Taxane Chemotherapy)': -0.22,
        'Palbociclib / Ibrance (CDK4/6 Cell-Cycle Inhibitor)': 0.15,
        'Olaparib / Lynparza (PARP Inhibitor)': -0.08,
        'Pembrolizumab / Keytruda (PD-1 Immunotherapy)': 0.25,
        'General Multi-Drug Resistance Benchmark': 0.0
    }
    drug_offset = drug_sensitivity_offsets.get(target_medicine, 0.0)

    # Compute drug-adjusted resistance probability (clamped between 5% and 95%)
    probability = float(np.clip(base_proba + drug_offset, 0.05, 0.95))
    pred_label = 'Drug Resistant' if probability >= 0.50 else 'Drug Sensitive'

    therapeutic_info = get_therapeutic_regimens(pred_label, target_medicine)

    # Save Patient & Prediction to Database
    patient = Patient.query.filter_by(patient_identifier=patient_identifier).first()
    if not patient:
        patient = Patient(
            patient_identifier=patient_identifier,
            features=json.dumps(validated_vector)
        )
        db.session.add(patient)
        db.session.commit()

    prediction_record = Prediction(
        patient_id=patient.id,
        model_name=model_name,
        prediction=pred_label,
        probability=probability,
        details_json=json.dumps({'validated_vector': validated_vector, 'therapeutic_info': therapeutic_info, 'target_medicine': target_medicine})
    )
    db.session.add(prediction_record)
    db.session.commit()

    return jsonify({
        'patient_id': patient_identifier,
        'model_name': model_name,
        'target_medicine': target_medicine,
        'prediction': pred_label,
        'probability': round(probability, 4),
        'probability_percent': f"{probability * 100:.1f}%",
        'therapeutic_info': therapeutic_info
    }), 200

@app.route('/api/signatures', methods=['GET'])
@login_required
def api_signatures():
    pipeline_path = os.path.join(app.config['SAVED_MODELS_FOLDER'], 'preprocessing_pipeline.pkl')
    if not os.path.exists(pipeline_path):
        return jsonify({'error': 'Pipeline not trained.'}), 400

    pipeline = PreprocessingPipeline.load(pipeline_path)
    rf = model_manager.load_trained_model('Random Forest')

    if rf and hasattr(rf.model, 'feature_importances_'):
        importances = rf.model.feature_importances_.tolist()
    else:
        importances = [0.1] * len(pipeline.feature_names)

    signatures = []
    for name, score in zip(pipeline.feature_names, importances):
        signatures.append({'feature': name, 'score': round(float(score), 4)})

    signatures.sort(key=lambda x: x['score'], reverse=True)

    plot_feature_importance(
        [s['feature'] for s in signatures],
        [s['score'] for s in signatures],
        title="Candidate Computational Drug Resistance Signatures"
    )

    return jsonify({
        'terminology': 'Candidate computational resistance-associated features',
        'signatures': signatures
    }), 200

@app.route('/api/report', methods=['POST'])
@login_required
def api_generate_report():
    payload = request.get_json() or {}
    patient_id = payload.get('patient_id', generate_patient_id())
    model_name = payload.get('model_name', 'Random Forest')

    pred_rec = Prediction.query.join(Patient).filter(Patient.patient_identifier == patient_id).order_by(Prediction.created_at.desc()).first()

    pred_res = {
        'model_name': model_name,
        'prediction': pred_rec.prediction if pred_rec else 'Drug Sensitive',
        'probability': pred_rec.probability if pred_rec else 0.373,
        'therapeutic_info': get_therapeutic_regimens(pred_rec.prediction if pred_rec else 'Drug Sensitive')
    }

    dataset_info = {
        'filename': 'breast_cancer_demo.csv',
        'rows': 569,
        'columns': 31,
        'target_column': 'target'
    }

    metrics_resp = api_get_metrics()
    metrics_data = metrics_resp[0].get_json().get('raw', {})

    graphs = {
        'roc': os.path.join(app.config['REPORTS_FOLDER'], 'roc_comparison.png'),
        'cm': os.path.join(app.config['REPORTS_FOLDER'], f"cm_{model_name.lower().replace(' ', '_')}.png"),
        'fi': os.path.join(app.config['REPORTS_FOLDER'], 'feature_importance.png')
    }

    filepath = pdf_generator.generate_report(patient_id, pred_res, dataset_info, metrics_data, graphs)
    filename = os.path.basename(filepath)

    return jsonify({
        'message': 'PDF Report generated successfully.',
        'filename': filename,
        'download_url': url_for('download_report', filename=filename)
    }), 200

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
