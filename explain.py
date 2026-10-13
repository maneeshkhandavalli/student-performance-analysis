"""SHAP explanations using XGBoost's built-in TreeSHAP (pred_contribs=True).

This gives exact SHAP values for tree models without needing the separate
`shap` package (which is heavy to install on Windows).
"""
import numpy as np
import pandas as pd
import xgboost as xgb


def shap_values(pipeline, X):
    """Return SHAP values with shape (n_samples, n_classes, n_features)."""
    booster = pipeline.named_steps["clf"].get_booster()
    contribs = booster.predict(xgb.DMatrix(pd.DataFrame(X).astype(float)), pred_contribs=True)
    return contribs[:, :, :-1]        # last column is the bias term


def explain_one(pipeline, X_row, class_idx, features, top=8):
    """SHAP values for one student and one class, sorted by absolute impact."""
    sv = shap_values(pipeline, X_row)[0, class_idx]
    pairs = sorted(((f, round(float(v), 3)) for f, v in zip(features, sv)), key=lambda kv: -abs(kv[1]))
    return pairs[:top]
