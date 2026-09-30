// Drug Resistance Prediction JavaScript Interface Handler

let currentPatientId = '';
let currentPredictionModel = '';

document.addEventListener('DOMContentLoaded', function() {
    initPredictionForm();
});

function initPredictionForm() {
    const btnGenId = document.getElementById('btnGenPatientId');
    const inputPatientId = document.getElementById('patient_id_input');
    const btnFillDemo = document.getElementById('btnFillDemoValues');
    const form = document.getElementById('predictionForm');
    const btnPDF = document.getElementById('btnGeneratePDF');

    if (btnGenId && inputPatientId) {
        btnGenId.addEventListener('click', () => {
            const randomId = 'PAT-' + Math.random().toString(36).substring(2, 8).toUpperCase();
            inputPatientId.value = randomId;
        });
    }

    // Auto load features schema
    loadFeatureInputSchema();

    if (btnFillDemo) {
        btnFillDemo.addEventListener('click', fillDemoFeatureValues);
    }

    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            executePrediction();
        });
    }

    if (btnPDF) {
        btnPDF.addEventListener('click', generatePDFReport);
    }
}

function loadFeatureInputSchema() {
    fetch('/api/dataset/summary')
        .then(res => res.json())
        .then(data => {
            const container = document.getElementById('featureInputContainer');
            if (!container) return;

            let cols = [];
            if (data.pipeline_features && data.pipeline_features.length > 0) {
                cols = data.pipeline_features;
            } else if (data.numerical_columns && data.numerical_columns.length > 0) {
                cols = data.numerical_columns.filter(c => c !== (data.target_column || 'target')).slice(0, 30);
            } else {
                cols = [
                    'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
                    'mean_smoothness', 'mean_compactness', 'mean_concavity',
                    'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
                    'radius_error', 'texture_error', 'perimeter_error', 'area_error',
                    'smoothness_error', 'compactness_error', 'concavity_error', 'concave_points_error',
                    'symmetry_error', 'fractal_dimension_error', 'worst_radius', 'worst_texture',
                    'worst_perimeter', 'worst_area', 'worst_smoothness', 'worst_compactness',
                    'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension'
                ];
            }

            container.innerHTML = cols.map(f => `
                <div class="row align-items-center mb-2">
                    <div class="col-7">
                        <label class="form-label small mb-0 fw-semibold text-truncate" title="${f}">${f}</label>
                    </div>
                    <div class="col-5">
                        <input type="number" step="any" class="form-control form-control-sm feature-val-input" data-feature="${f}" placeholder="0.0">
                    </div>
                </div>
            `).join('');
        })
        .catch(err => console.error(err));
}

function fillDemoFeatureValues() {
    const inputs = document.querySelectorAll('.feature-val-input');
    inputs.forEach((input, index) => {
        // Generate realistic dummy values around standard scale mean
        const val = (Math.sin(index) * 1.5 + 1.0).toFixed(4);
        input.value = val;
    });
}

function executePrediction() {
    const btnPredict = document.getElementById('btnPredict');
    const inputPatientId = document.getElementById('patient_id_input');
    const modelSelect = document.getElementById('model_select');
    const medicineSelect = document.getElementById('medicine_select');

    const patientId = inputPatientId.value.trim() || ('PAT-' + Math.random().toString(36).substring(2, 8).toUpperCase());
    inputPatientId.value = patientId;

    const modelName = modelSelect.value;
    const targetMedicine = medicineSelect ? medicineSelect.value : 'General Multi-Drug Resistance Benchmark';
    const inputs = document.querySelectorAll('.feature-val-input');
    const features = {};

    inputs.forEach(input => {
        const featName = input.getAttribute('data-feature');
        const val = parseFloat(input.value) || 0.0;
        features[featName] = val;
    });

    btnPredict.disabled = true;
    btnPredict.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></i> Predicting...';

    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            patient_id: patientId,
            model_name: modelName,
            target_medicine: targetMedicine,
            features: features
        })
    })
    .then(res => res.json())
    .then(data => {
        btnPredict.disabled = false;
        btnPredict.innerHTML = '<i class="bi bi-cpu me-1"></i> Generate Drug Resistance Prediction';

        if (data.error) {
            alert(data.error);
            return;
        }

        currentPatientId = data.patient_id;
        currentPredictionModel = data.model_name;

        document.getElementById('predictionPlaceholder').classList.add('d-none');
        const resContent = document.getElementById('predictionResultContent');
        resContent.classList.remove('d-none');

        const labelEl = document.getElementById('resPredictionLabel');
        labelEl.innerText = data.prediction;
        if (data.prediction === 'Drug Resistant') {
            labelEl.className = 'fw-bold mb-3 text-danger';
        } else {
            labelEl.className = 'fw-bold mb-3 text-success';
        }

        document.getElementById('resProbabilityScore').innerText = data.probability_percent;
        document.getElementById('resModelBadge').innerText = data.model_name;

        if (data.therapeutic_info) {
            const tContainer = document.getElementById('therapeuticInfoContainer');
            tContainer.classList.remove('d-none');
            document.getElementById('resTherapeuticCategory').innerText = data.therapeutic_info.category || '-';
            document.getElementById('resMechanismSummary').innerText = data.therapeutic_info.mechanism_summary || '-';

            const badgeBox = document.getElementById('resMedicineBadges');
            const medList = data.therapeutic_info.medicines || [];
            badgeBox.innerHTML = medList.map(m => `
                <div class="badge bg-light text-dark border p-2 text-start w-100 mb-1" style="font-size: 0.78rem;">
                    <i class="bi bi-prescription text-primary me-1"></i><strong>${m.name}</strong>
                    <span class="text-muted d-block ms-3">Class: ${m.class} | Target: ${m.target}</span>
                </div>
            `).join('');
        }
    })
    .catch(err => {
        btnPredict.disabled = false;
        btnPredict.innerHTML = '<i class="bi bi-cpu me-1"></i> Generate Drug Resistance Prediction';
        console.error(err);
    });
}

function generatePDFReport() {
    if (!currentPatientId) {
        alert('Please generate a prediction first.');
        return;
    }

    fetch('/api/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            patient_id: currentPatientId,
            model_name: currentPredictionModel
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
        } else {
            alert('PDF Report generated successfully! Redirecting to download...');
            window.location.href = data.download_url;
        }
    })
    .catch(err => console.error(err));
}
