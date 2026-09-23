"""PATCH: XGBoost fallback for ML classification (stage 5).

Random Forest is attempted first — identical hyperparameters to the
original train_model().  Only if RF raises any exception does this patch
transparently retry with XGBClassifier.

Both classifiers expose the same predict_proba(X) interface so nothing
downstream (run_ml_classification, ensemble scoring, etc.) needs to change.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _patched_train_model(X: np.ndarray, y: np.ndarray):
    """Train Random Forest; fall back to XGBoost on any RF exception.

    Returns a fitted model that exposes ``predict_proba(X) -> array``.
    The returned type is either ``RandomForestClassifier`` or
    ``XGBClassifier`` — both satisfy the interface expected by
    :func:`tess_pipeline.stage5_classify.run_ml_classification`.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from tess_pipeline.stage5_classify import PHYSICS_FEATURES

    # --- Single-class edge case (unchanged from original) ---
    if len(np.unique(y)) < 2:
        print("  ⚠️  Only one class. Using dummy model.")
        dummy = RandomForestClassifier(n_estimators=10, random_state=42)
        dummy.fit(X, y)
        return dummy

    # --- Compute class weights (unchanged from original) ---
    n_planets = int(sum(y == 1))
    n_non = int(sum(y == 0))
    if n_planets < n_non:
        class_weight = {0: 1.0, 1: n_non / max(n_planets, 1)}
    elif n_non < n_planets:
        class_weight = {0: n_planets / max(n_non, 1), 1: 1.0}
    else:
        class_weight = "balanced"

    # =========================================================
    # Attempt 1 — Random Forest (exact original behaviour)
    # =========================================================
    try:
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=5,
            min_samples_leaf=3, max_features="sqrt",
            class_weight=class_weight, random_state=42, n_jobs=-1,
        )
        if len(y) >= 10:
            cv = StratifiedKFold(
                n_splits=min(5, len(y) // 2), shuffle=True, random_state=42
            )
            scores = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
            print(f"\n  🔬 CV AUC: {scores.mean():.3f} (+/- {scores.std():.3f})")
        rf.fit(X, y)
        print(f"\n  📈 Feature Importance:")
        for feat, imp in zip(PHYSICS_FEATURES, rf.feature_importances_):
            bar = "█" * int(imp * 30)
            print(f"     {feat:<20} {imp:.3f} {bar}")
        return rf

    except Exception as rf_err:
        logger.warning(
            "Random Forest training failed (%s) — switching to XGBoost fallback.",
            rf_err,
        )
        print(f"\n  ⚠️  Random Forest failed: {rf_err}")
        print("  🔄 Falling back to XGBoost …")

    # =========================================================
    # Attempt 2 — XGBoost fallback
    # =========================================================
    from xgboost import XGBClassifier

    scale_pos_weight = n_non / max(n_planets, 1) if n_planets > 0 else 1.0
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    if len(y) >= 10:
        cv = StratifiedKFold(
            n_splits=min(5, len(y) // 2), shuffle=True, random_state=42
        )
        scores = cross_val_score(xgb, X, y, cv=cv, scoring="roc_auc")
        print(f"\n  🔬 XGB CV AUC: {scores.mean():.3f} (+/- {scores.std():.3f})")
    xgb.fit(X, y)
    print(f"\n  📈 Feature Importance (XGBoost):")
    importances = xgb.feature_importances_
    for feat, imp in zip(PHYSICS_FEATURES, importances):
        bar = "█" * int(imp * 30)
        print(f"     {feat:<20} {imp:.3f} {bar}")
    print("\n  ✅ XGBoost fallback model trained successfully.")
    return xgb


def apply():
    """Monkey-patch stage5_classify.train_model with RF + XGBoost fallback."""
    import tess_pipeline.stage5_classify as s5
    s5.train_model = _patched_train_model
    print("  ✅ PATCH xgboost_fallback: train_model → RF with XGBoost fallback")
