# Predictive Modeling in Oncology Using Quantum Boltzmann Machines for Rapid Identification of Drug Resistance Signatures

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.35-purple.svg)](https://pennylane.ai/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0-blueviolet.svg)](https://qiskit.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

> **RESEARCH PROTOTYPE DISCLAIMER**  
> *This software is a research prototype intended for computational experimentation and academic demonstration. It is NOT a medical diagnostic system and must NOT be used independently to make clinical treatment decisions.*

---

## 1. PROJECT OVERVIEW

This application is an end-to-end full-stack oncology research and decision-support prototype designed to evaluate classical machine learning models (SVM, Random Forest, Deep Learning) alongside Quantum-Inspired Variational Boltzmann Machines (QBM) for predicting cancer drug resistance from high-dimensional genomic features.

### Primary Objectives
1. Authenticate researchers and manage anonymized patient data.
2. Upload, validate, and preprocess genomic CSV datasets.
3. Train classical machine learning models (SVM, Random Forest, TensorFlow MLP).
4. Train variational quantum circuits and QBM-inspired models using PennyLane / Qiskit Aer CPU simulation.
5. Benchmark and compare classical vs quantum model performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Training Time).
6. Explain model predictions with SHAP and Permutation Feature Importance.
7. Identify candidate computational drug resistance feature signatures.
8. Generate and download automated ReportLab PDF research summaries.

---

## 2. TECHNOLOGY STACK

- **Backend Framework**: Python 3.11, Flask, Flask-SQLAlchemy, Flask-Login, Werkzeug
- **Database**: SQLite (SQLAlchemy ORM)
- **Machine Learning**: NumPy, Pandas, Scikit-Learn, TensorFlow / Keras, Joblib
- **Quantum Machine Learning**: PennyLane, Qiskit, Qiskit Aer Simulator
- **Visualization**: Matplotlib, Seaborn, Chart.js
- **Explainability**: SHAP (TreeExplainer / Permutation Explainer)
- **PDF Report Generation**: ReportLab
- **Frontend**: HTML5, CSS3, JavaScript (ES6+), Bootstrap 5, Bootstrap Icons
- **Automated Testing**: Pytest

---

## 3. FOLDER STRUCTURE

```text
Predictive_Oncology/
│
├── app.py                      # Main Flask Web Application Server & API Routes
├── config.py                   # Environment & Application Configuration
├── requirements.txt            # Python Dependencies Specification
├── README.md                   # Complete Documentation & Usage Guide
├── PROJECT_REPORT.md           # Comprehensive Research Project Report
├── .env.example                # Template for Environment Variables
├── .gitignore                  # Git Ignore Specifications
├── database.db                 # SQLite Database File
│
├── database/                   # Database Models & Initialization
│   ├── __init__.py
│   ├── models.py               # SQLAlchemy ORM Models (User, Patient, etc.)
│   └── init_db.py              # Schema Setup & Demo Seed Script
│
├── dataset/                    # Raw, Processed, and Sample Datasets
│   ├── raw/
│   ├── processed/
│   └── sample/                 # Breast Cancer Wisconsin Demo CSV
│
├── static/                     # Web Static Assets
│   ├── css/
│   │   └── style.css           # Custom Medical Research CSS Theme
│   └── js/
│       ├── dashboard.js        # Dashboard Chart.js Script
│       ├── charts.js           # Model Training Event Handlers
│       └── prediction.js       # Patient Prediction & PDF Download Handler
│
├── templates/                  # Jinja2 HTML Templates
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── upload.html
│   ├── preprocessing.html
│   ├── models.html
│   ├── quantum.html
│   ├── prediction.html
│   ├── explainability.html
│   ├── reports.html
│   └── error.html
│
├── models/                     # Classical & Quantum Model Wrappers
│   ├── __init__.py
│   ├── svm_model.py            # Support Vector Machine Wrapper
│   ├── random_forest.py        # Random Forest Wrapper
│   ├── deep_learning.py        # TensorFlow/Keras Neural Network
│   ├── quantum_model.py        # Quantum Model Wrapper
│   └── model_manager.py        # Central Orchestrator
│
├── preprocessing/              # Data Pipeline Modules
│   ├── __init__.py
│   ├── cleaner.py              # Missing Value Imputer & Duplicate Cleaner
│   ├── encoder.py              # Categorical & Target Label Encoder
│   ├── scaler.py               # StandardScaler (Zero Leakage)
│   ├── feature_selection.py    # Dimensionality Reduction Engine
│   └── pipeline.py             # End-to-End Preprocessing Pipeline
│
├── quantum/                    # QML & Circuit Simulation Engine
│   ├── __init__.py
│   ├── circuits.py             # Variational Quantum Circuit (VQC)
│   ├── embedding.py            # Angle Embedding Rotations
│   ├── qbm.py                  # Quantum-Inspired Boltzmann Machine
│   └── simulator.py            # Simulator Backend Manager
│
├── utils/                      # Auxiliary Helper Modules
│   ├── __init__.py
│   ├── validators.py           # Input & File Validator
│   ├── security.py             # Password Hashing & Sanitization
│   ├── metrics.py              # Performance Metrics Calculator
│   ├── plotting.py             # Matplotlib/Seaborn Chart Exporter
│   ├── explainability.py       # SHAP / Permutation Importance Engine
│   └── helpers.py              # Timer & Serialization Helpers
│
├── reports/                    # PDF Report Generator
│   ├── generated/              # Output PDF Files
│   └── report_generator.py     # ReportLab PDF Engine
│
├── saved_models/               # Serialized Models (.pkl, .keras)
└── tests/                      # Pytest Automated Test Suite
    ├── test_auth.py
    ├── test_preprocessing.py
    ├── test_models.py
    ├── test_quantum.py
    ├── test_prediction.py
    └── test_routes.py
```

---

## 4. INSTALLATION & SETUP (WINDOWS POWERSHELL)

Follow these exact steps to set up and run the application locally on Windows 11 / Windows 10:

### Step 1: Open Windows PowerShell and navigate to workspace
```powershell
cd c:\antigravity\oncology
```

### Step 2: Create Python 3.11 Virtual Environment
```powershell
python -m venv .venv
```

### Step 3: Activate Virtual Environment
```powershell
.venv\Scripts\activate
```

### Step 4: Upgrade Pip and Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Initialize Database and Seed Demo Data
```powershell
python database/init_db.py
```

### Step 6: Start Flask Web Server
```powershell
python app.py
```

### Step 7: Access Web Application
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

Default Admin Researcher Account:
- **Username**: `researcher`
- **Password**: `ResearchPass2026!`

---

## 5. AUTOMATED TESTING

Execute the pytest suite to verify all system modules:
```powershell
pytest -v
```

---

## 6. REST API DOCUMENTATION

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/health` | `GET` | Health check endpoint returning API status |
| `/api/dataset/summary` | `GET` | Returns dataset row/col counts, distribution & preview |
| `/api/train/<model_type>` | `POST` | Trains requested model (`svm`, `random-forest`, `deep-learning`, `quantum`) |
| `/api/metrics` | `GET` | Returns benchmark performance summary table and metrics |
| `/api/predict` | `POST` | Generates patient drug resistance prediction & probability score |
| `/api/signatures` | `GET` | Returns candidate computational resistance feature signatures |
| `/api/report` | `POST` | Generates ReportLab PDF research summary report |

---

## 7. SCIENTIFIC & QUANTUM LIMITATIONS

1. **Simulated Quantum Hardware**: The QBM module utilizes local classical CPU statevector simulation (PennyLane / Qiskit Aer). No claims of physical quantum supremacy or hardware advantage are asserted.
2. **Computational Feature Signatures**: Identified predictive features represent mathematical correlations in benchmark datasets and must be independently validated before biological biomarker designation.

---

## 8. LICENSE & ACKNOWLEDGMENTS

Developed for academic research and final-year B.Tech capstone demonstration. Released under the MIT License.
