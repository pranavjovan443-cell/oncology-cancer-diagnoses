// Dashboard JavaScript Chart Renderers with HealMind AI Brand Palette

document.addEventListener('DOMContentLoaded', function() {
    initPredictionDistChart();
    initModelMetricsChart();
});

function initPredictionDistChart() {
    const ctx = document.getElementById('predictionDistChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Drug Resistant', 'Drug Sensitive'],
            datasets: [{
                data: [65, 35],
                backgroundColor: ['#ef4444', '#10b981'],
                borderWidth: 3,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { 
                    position: 'bottom',
                    labels: {
                        usePointStyle: true,
                        padding: 20,
                        font: { family: 'Plus Jakarta Sans', size: 12, weight: '600' }
                    }
                }
            },
            cutout: '70%'
        }
    });
}

function initModelMetricsChart() {
    const ctx = document.getElementById('modelMetricsChart');
    if (!ctx) return;

    fetch('/api/metrics')
        .then(res => res.json())
        .then(data => {
            const summary = data.summary || [];
            const labels = summary.map(item => item.Model);
            const accuracies = summary.map(item => (item.Accuracy * 100).toFixed(1));

            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels.length ? labels : ['Quantum Model (QBM)'],
                    datasets: [{
                        label: 'Accuracy (%)',
                        data: accuracies.length ? accuracies : [85.0],
                        backgroundColor: ['#4f46e5', '#10b981', '#ef4444', '#f59e0b'],
                        borderRadius: 8
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: { 
                            min: 0, 
                            max: 100,
                            grid: { color: '#eaeff6' },
                            ticks: { font: { family: 'Plus Jakarta Sans', size: 11 } }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { font: { family: 'Plus Jakarta Sans', size: 11, weight: '600' } }
                        }
                    }
                }
            });
        })
        .catch(err => console.error(err));
}
