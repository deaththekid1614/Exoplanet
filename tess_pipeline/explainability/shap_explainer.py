"""SHAP explainability for XGBoost predictions."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def explain_xgboost_prediction(model, features_df: pd.DataFrame,
                               feature_names: list[str] | None = None) -> dict:
    """Generate SHAP explanation for a single XGBoost prediction.
    
    Returns SHAP values with terminal-friendly summary.
    """
    print(f"    ┌─ XGBoost Explainability (SHAP) ─────────────────┐")
    
    try:
        import shap
    except ImportError:
        print(f"    │  ⚠️  SHAP not installed. Run: pip install shap")
        print(f"    └─────────────────────────────────────────────────┘")
        return {"shap_available": False, "shap_values": None}
    
    # Ensure features are numeric
    X = features_df.select_dtypes(include=[np.number]).values
    if feature_names is None:
        feature_names = list(features_df.select_dtypes(include=[np.number]).columns)
    
    # Create explainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Handle multi-class (list of arrays) vs binary (single array)
        if isinstance(shap_values, list):
            # Multi-class: take positive class (index 1)
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            sv = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        
        # Top contributing features
        top_indices = np.argsort(np.abs(sv))[-5:][::-1]
        
        result = {
            "shap_available": True,
            "shap_values": sv.tolist(),
            "top_positive_features": [],
            "top_negative_features": [],
            "base_value": float(explainer.expected_value[1]) if isinstance(explainer.expected_value, list) else float(explainer.expected_value),
        }
        
        print(f"    │  Base Value: {result['base_value']:.4f}")
        print(f"    │  Top 5 Feature Contributions:")
        
        for idx in top_indices:
            fname = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
            val = float(sv[idx])
            direction = "↑" if val > 0 else "↓"
            print(f"    │    {direction} {fname:20s}: {val:+.4f}")
            
            if val > 0:
                result["top_positive_features"].append({"feature": fname, "impact": val})
            else:
                result["top_negative_features"].append({"feature": fname, "impact": val})
        
        print(f"    └─────────────────────────────────────────────────┘")
        
        logger.info("  [SHAP] Explained %d features, base_value=%.4f",
                    len(feature_names), result["base_value"])
        
        return result
        
    except Exception as e:
        logger.warning("SHAP explanation failed: %s", e)
        print(f"    │  ⚠️  SHAP failed: {e}")
        print(f"    └─────────────────────────────────────────────────┘")
        return {"shap_available": False, "shap_values": None, "error": str(e)}
