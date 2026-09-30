import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    balanced_accuracy_score, matthews_corrcoef, average_precision_score
)

def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, Any]:
    """
    Compute full suite of classification performance metrics.
    
    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
        y_proba: Predicted probability scores for positive class (0.0 to 1.0)
        
    Returns:
        Dictionary of performance metrics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    else:
        tn = fp = fn = tp = 0
        specificity = 0.0

    # ROC AUC & Curves
    roc_auc = 0.5
    fpr_list, tpr_list, roc_thresholds = [], [], []
    precision_list, recall_list, pr_thresholds = [], [], []
    avg_precision = prec

    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_proba))
            fpr, tpr, roc_thresh = roc_curve(y_true, y_proba)
            fpr_list = fpr.tolist()
            tpr_list = tpr.tolist()
            roc_thresholds = roc_thresh.tolist()

            prec_curve, rec_curve, pr_thresh = precision_recall_curve(y_true, y_proba)
            precision_list = prec_curve.tolist()
            recall_list = rec_curve.tolist()
            pr_thresholds = pr_thresh.tolist()
            avg_precision = float(average_precision_score(y_true, y_proba))
        except Exception:
            roc_auc = 0.5

    metrics = {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc_auc, 4),
        'specificity': round(specificity, 4),
        'balanced_accuracy': round(bal_acc, 4),
        'mcc': round(mcc, 4),
        'average_precision': round(avg_precision, 4),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'raw': cm.tolist()
        },
        'roc_curve': {
            'fpr': fpr_list,
            'tpr': tpr_list
        },
        'pr_curve': {
            'precision': precision_list,
            'recall': recall_list
        }
    }

    return metrics
