import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.inspection import permutation_importance

class ModelExplainer:
    """
    Explainability engine integrating SHAP with robust Permutation Importance fallback.
    Explicitly tags method used.
    """

    def __init__(self, model: Any, X_train: np.ndarray, feature_names: List[str]):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.method_used = "Permutation Importance"

    def compute_global_importance(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Compute global feature importances using SHAP when feasible, falling back to Permutation Importance.
        """
        try:
            import shap
            # Try SHAP TreeExplainer for Tree models
            if hasattr(self.model, 'feature_importances_'):
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_test)
                
                # Handle binary classification 2D or 3D shap output
                if isinstance(shap_values, list):
                    values = np.abs(shap_values[1]).mean(axis=0)
                elif isinstance(shap_values, np.ndarray):
                    if shap_values.ndim == 3:
                        values = np.abs(shap_values[:, :, 1]).mean(axis=0)
                    else:
                        values = np.abs(shap_values).mean(axis=0)
                else:
                    values = np.abs(shap_values).mean(axis=0)
                
                self.method_used = "SHAP TreeExplainer"
                importance_dict = dict(zip(self.feature_names, values.tolist()))
                return {
                    'method': self.method_used,
                    'importances': sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                }
        except Exception as e:
            # Fallback to Permutation Importance if SHAP errors
            pass

        # Permutation Importance Fallback
        res = permutation_importance(self.model, X_test, y_test, n_repeats=5, random_state=42)
        importances = res.importances_mean
        importance_dict = dict(zip(self.feature_names, importances.tolist()))
        self.method_used = "Permutation Importance (Fallback)"

        return {
            'method': self.method_used,
            'importances': sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        }

    def explain_single_instance(self, instance_vector: np.ndarray) -> Dict[str, Any]:
        """
        Explain single patient prediction instance feature contribution.
        """
        instance_vector = np.array(instance_vector).reshape(1, -1)
        try:
            import shap
            if hasattr(self.model, 'feature_importances_'):
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(instance_vector)
                
                if isinstance(shap_values, list):
                    vals = shap_values[1][0]
                elif isinstance(shap_values, np.ndarray):
                    if shap_values.ndim == 3:
                        vals = shap_values[0, :, 1]
                    else:
                        vals = shap_values[0]
                else:
                    vals = shap_values[0]

                contributions = []
                for name, val, impact in zip(self.feature_names, instance_vector[0], vals):
                    contributions.append({
                        'feature': name,
                        'value': round(float(val), 4),
                        'contribution': round(float(impact), 4),
                        'direction': 'Promotes Resistance' if impact > 0 else 'Promotes Sensitivity'
                    })

                contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
                return {
                    'method': "SHAP Individual Attribution",
                    'contributions': contributions
                }
        except Exception:
            pass

        # Feature perturbation attribution fallback
        baseline_pred = self.model.predict_proba(instance_vector)[0][1] if hasattr(self.model, 'predict_proba') else self.model.predict(instance_vector)[0]
        contributions = []
        
        for i, (name, val) in enumerate(zip(self.feature_names, instance_vector[0])):
            perturbed = instance_vector.copy()
            perturbed[0, i] = 0.0  # zero out feature
            if hasattr(self.model, 'predict_proba'):
                new_pred = self.model.predict_proba(perturbed)[0][1]
            else:
                new_pred = self.model.predict(perturbed)[0]
            
            diff = float(baseline_pred - new_pred)
            contributions.append({
                'feature': name,
                'value': round(float(val), 4),
                'contribution': round(diff, 4),
                'direction': 'Promotes Resistance' if diff > 0 else 'Promotes Sensitivity'
            })

        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        return {
            'method': "Feature Perturbation Attribution",
            'contributions': contributions
        }
