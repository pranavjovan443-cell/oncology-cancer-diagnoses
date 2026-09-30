// Classical Models Page JavaScript Handler

document.addEventListener('DOMContentLoaded', function() {
    setupModelTrainButtons();
    loadMetricsTable();
});

function setupModelTrainButtons() {
    const btnSVM = document.getElementById('btnTrainSVM');
    const btnRF = document.getElementById('btnTrainRF');
    const btnDL = document.getElementById('btnTrainDL');
    const btnRefresh = document.getElementById('btnRefreshMetrics');

    if (btnSVM) btnSVM.addEventListener('click', () => trainModel('svm', btnSVM));
    if (btnRF) btnRF.addEventListener('click', () => trainModel('random-forest', btnRF));
    if (btnDL) btnDL.addEventListener('click', () => trainModel('deep-learning', btnDL));
    if (btnRefresh) btnRefresh.addEventListener('click', loadMetricsTable);
}

function trainModel(modelType, btnElement) {
    const originalText = btnElement.innerHTML;
    btnElement.disabled = true;
    btnElement.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></i> Training...';

    fetch(`/api/train/${modelType}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(res => res.json())
    .then(data => {
        btnElement.disabled = false;
        btnElement.innerHTML = originalText;
        if (data.error) {
            alert(data.error);
        } else {
            alert(data.message || 'Model trained successfully.');
            loadMetricsTable();
        }
    })
    .catch(err => {
        btnElement.disabled = false;
        btnElement.innerHTML = originalText;
        console.error(err);
    });
}

function loadMetricsTable() {
    fetch('/api/metrics')
        .then(res => res.json())
        .then(data => {
            const summary = data.summary || [];
            const tbody = document.getElementById('metricsTableBody');
            if (!tbody) return;

            if (summary.length === 0) {
                tbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted py-4">No models trained yet. Click "Train" on a model above.</td></tr>`;
                return;
            }

            tbody.innerHTML = summary.map(item => `
                <tr>
                    <td class="fw-bold">${item.Model}</td>
                    <td><span class="badge bg-primary fs-6">${(item.Accuracy * 100).toFixed(1)}%</span></td>
                    <td>${(item.Precision * 100).toFixed(1)}%</td>
                    <td>${(item.Recall * 100).toFixed(1)}%</td>
                    <td>${(item.F1 * 100).toFixed(1)}%</td>
                    <td>${item['ROC-AUC'].toFixed(3)}</td>
                    <td class="text-muted">${item['Training Time']}</td>
                </tr>
            `).join('');
        })
        .catch(err => console.error(err));
}
