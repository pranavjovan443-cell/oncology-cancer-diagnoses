import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server generation
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from config import Config

# Set global publication plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.sans-serif': 'Helvetica, Arial, sans-serif',
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'grid.color': '#eeeeee'
})

def plot_confusion_matrix(cm_data: list, model_name: str, output_dir: str = None) -> str:
    """Generate and save Seaborn confusion matrix heatmap."""
    output_dir = output_dir or Config.REPORTS_FOLDER
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    cm = np.array(cm_data)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Sensitive (0)', 'Resistant (1)'],
                yticklabels=['Sensitive (0)', 'Resistant (1)'], ax=ax)
    
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=10, fontweight='bold')
    plt.tight_layout()
    
    filename = f"cm_{model_name.lower().replace(' ', '_')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return filepath

def plot_roc_curves(models_metrics: Dict[str, Dict[str, Any]], output_dir: str = None) -> str:
    """Generate combined ROC curve plot for all trained models."""
    output_dir = output_dir or Config.REPORTS_FOLDER
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    colors = {'SVM': '#2b5c8f', 'Random Forest': '#27ae60', 'Deep Learning': '#e74c3c', 'Quantum Model': '#8e44ad', 'QBM-Inspired': '#d35400'}

    for model_name, metrics in models_metrics.items():
        if 'roc_curve' in metrics and metrics['roc_curve'].get('fpr'):
            fpr = metrics['roc_curve']['fpr']
            tpr = metrics['roc_curve']['tpr']
            auc = metrics.get('roc_auc', 0.5)
            color = colors.get(model_name, '#333333')
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.500)', linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=10, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Comparison', fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'roc_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return filepath

def plot_precision_recall_curves(models_metrics: Dict[str, Dict[str, Any]], output_dir: str = None) -> str:
    """Generate combined Precision-Recall curve comparison."""
    output_dir = output_dir or Config.REPORTS_FOLDER
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    colors = {'SVM': '#2b5c8f', 'Random Forest': '#27ae60', 'Deep Learning': '#e74c3c', 'Quantum Model': '#8e44ad', 'QBM-Inspired': '#d35400'}

    for model_name, metrics in models_metrics.items():
        if 'pr_curve' in metrics and metrics['pr_curve'].get('precision'):
            prec = metrics['pr_curve']['precision']
            rec = metrics['pr_curve']['recall']
            ap = metrics.get('average_precision', 0.0)
            color = colors.get(model_name, '#333333')
            ax.plot(rec, prec, label=f'{model_name} (AP = {ap:.3f})', color=color, linewidth=2)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Sensitivity)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=10, fontweight='bold')
    ax.set_title('Precision-Recall Curves Comparison', fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'pr_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return filepath

def plot_feature_importance(feature_names: List[str], importance_scores: List[float], title: str = "Feature Importance", output_dir: str = None) -> str:
    """Generate horizontal bar chart for top predictive feature signatures."""
    output_dir = output_dir or Config.REPORTS_FOLDER
    os.makedirs(output_dir, exist_ok=True)

    # Sort features
    pairs = sorted(zip(feature_names, importance_scores), key=lambda x: x[1], reverse=True)[:15]
    names = [p[0] for p in reversed(pairs)]
    scores = [p[1] for p in reversed(pairs)]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.barh(names, scores, color='#1f77b4', edgecolor='#0d47a1', height=0.6)
    ax.set_xlabel('Predictive Importance Score', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return filepath

def plot_model_comparison_bar(models_metrics: Dict[str, Dict[str, Any]], output_dir: str = None) -> str:
    """Generate side-by-side performance metrics benchmark chart."""
    output_dir = output_dir or Config.REPORTS_FOLDER
    os.makedirs(output_dir, exist_ok=True)

    models = list(models_metrics.keys())
    metrics_list = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

    data = {m: [models_metrics[m].get(metric, 0) for metric in metrics_list] for m in models}
    df = pd.DataFrame(data, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    df.plot(kind='bar', ax=ax, width=0.75, colormap='viridis')
    ax.set_ylim([0, 1.1])
    ax.set_ylabel('Score', fontsize=10, fontweight='bold')
    ax.set_title('Classical vs Quantum Model Metrics Benchmark', fontsize=12, fontweight='bold', pad=10)
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'model_comparison_bar.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return filepath

def plot_training_loss(loss_history: List[float], val_loss_history: Optional[List[float]] = None, title: str = "Training Loss History", output_dir: str = None) -> str:
    """Plot convergence optimization loss curves."""
    output_dir = output_dir or Config.REPORTS_FOLDER
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    epochs = range(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, 'b-o', label='Training Loss', linewidth=2, markersize=4)
    if val_loss_history:
        ax.plot(epochs, val_loss_history, 'r--s', label='Validation Loss', linewidth=2, markersize=4)

    ax.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.legend()
    plt.tight_layout()

    filename = f"loss_{title.lower().replace(' ', '_')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return filepath
