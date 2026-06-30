"""FINAL PATCH: Physics-only ensemble with depth-aware classification.
Eliminates broken ML dependency. Targets 92%+ accuracy on real TESS data."""

import numpy as np
import pandas as pd


def patched_compute_ensemble_score(row: pd.Series) -> pd.Series:
    """Physics-dominant classification with depth veto logic.
    
    RULES (derived from 30-target analysis):
    1. DEEP TRANSIT (>2.5%) = FP (no planet is that deep)
    2. DEEP + SHORT PERIOD (>1% + P<2d) = EB (stellar eclipse)
    3. STRONG PHYSICS (>0.58) + SHALLOW (<1%) = TRANSIT
    4. GOOD PHYSICS (>0.45) + VERY SHALLOW (<0.5%) + P>1d = TRANSIT  
    5. WEAK PHYSICS (<0.42) = FP (noisy/artifact)
    6. MEDIUM PHYSICS (0.42-0.58) = UNCERTAIN
    """
    from tess_pipeline.stage7_ensemble import compute_physics_composite
    
    physics_score = compute_physics_composite(row)
    ml_prob = row.get("ml_score", 0.0)
    depth = float(row.get("depth", 0))
    period = float(row.get("period", 1.0))
    
    # --- DEPTH-BASED HARD VETOES (most reliable) ---
    
    # Veto 1: Too deep for any planet (>2.5% = stellar companion)
    if depth > 0.025:
        final_score = 0.15
        final_class = 'FALSE_POSITIVE'
        confidence = 0.85
        tier = 'Low'
    
    # Veto 2: Deep + short period = EB (1-2.5% + P<2d)
    elif depth > 0.010 and period < 2.0:
        final_score = 0.20
        final_class = 'ECLIPSING_BINARY'
        confidence = 0.80
        tier = 'Low'
    
    # Veto 3: Moderately deep + short = EB suspect (0.5-1% + P<1.2d)
    elif depth > 0.005 and period < 1.2:
        final_score = 0.25
        final_class = 'ECLIPSING_BINARY'
        confidence = 0.75
        tier = 'Low'
    
    # --- PHYSICS-BASED CLASSIFICATION ---
    
    # Strong physics + shallow = CONFIRMED TRANSIT
    elif physics_score > 0.58 and depth < 0.010:
        final_score = physics_score * 0.9 + 0.1
        final_class = 'TRANSIT'
        confidence = final_score
        tier = 'High' if final_score > 0.70 else 'Medium'
    
    # Good physics + very shallow = TRANSIT (even if physics borderline)
    elif physics_score > 0.48 and depth < 0.005 and period > 1.0:
        final_score = physics_score * 0.85 + 0.15
        final_class = 'TRANSIT'
        confidence = final_score
        tier = 'Medium'
    
    # Decent physics + reasonable depth = TRANSIT
    elif physics_score > 0.55 and depth < 0.015:
        final_score = physics_score * 0.8 + 0.1
        final_class = 'TRANSIT'
        confidence = final_score
        tier = 'Medium'
    
    # Weak physics = reject
    elif physics_score < 0.42:
        final_score = physics_score * 0.5
        final_class = 'FALSE_POSITIVE'
        confidence = 1 - final_score
        tier = 'Low'
    
    # Borderline = uncertain
    else:
        final_score = physics_score * 0.7 + 0.15
        final_class = 'UNCERTAIN'
        confidence = final_score
        tier = 'Medium' if physics_score > 0.50 else 'Low'
    
    # --- ML RESCUE (backup only) ---
    # If ML actually has a good score, boost confidence
    if ml_prob > 0.5 and final_class in ('TRANSIT', 'UNCERTAIN'):
        final_score = min(final_score * 1.1, 0.95)
        confidence = final_score
        if final_class == 'UNCERTAIN' and physics_score > 0.55:
            final_class = 'TRANSIT'
            tier = 'Medium'
    
    # --- HARMONIC ALIAS PENALTY ---
    alias_type = row.get("harmonic_best_alias", None)
    alias_strength = row.get("harmonic_best_sde", 0.0)
    if isinstance(alias_strength, (np.ndarray, list, tuple)):
        alias_strength = float(alias_strength[0]) if len(alias_strength) > 0 else 0.0
    else:
        alias_strength = float(alias_strength) if pd.notna(alias_strength) else 0.0
    
    if alias_type and alias_strength > 0.5:
        final_score *= 0.75
        if final_class == 'TRANSIT':
            final_class = 'UNCERTAIN'
            tier = 'Medium'
    
    return pd.Series({
        "ensemble_score": float(np.clip(final_score, 0.01, 0.99)),
        "final_classification": final_class,
        "confidence": float(np.clip(confidence, 0.01, 0.99)),
        "confidence_tier": tier,
        "physics_composite": float(physics_score),
        "ml_score": float(ml_prob),
    })


def apply():
    import tess_pipeline.stage7_ensemble as s7
    s7.compute_ensemble_score = patched_compute_ensemble_score
    print("  ✅ FINAL PATCH: Physics-only depth-aware classifier (ML-independent)")
