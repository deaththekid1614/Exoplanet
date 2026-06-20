"""Stage 7: Ensemble Scoring — Physics (60%) + ML (40%)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np # type: ignore
import pandas as pd # type: ignore

logger = logging.getLogger(__name__)

PHYSICS_WEIGHT = 0.60
ML_WEIGHT = 0.40


def compute_physics_composite(row: pd.Series) -> float:
    """Compute weighted physics composite score from a row."""
    depth_score = row.get("depth_score", 0.5)
    shape_score = row.get("shape_score", 0.5)
    nosec_score = row.get("nosec_score", 0.5)
    oddeven_score = row.get("oddeven_score", 0.5)
    achrom_score = row.get("achrom_score", 0.5)
    atmos_score = row.get("atmos_score", 0.5)
    det_quality = row.get("det_quality", 0.5)

    composite = (
        det_quality * 0.10 +
        depth_score * 0.20 +
        shape_score * 0.15 +
        nosec_score * 0.15 +
        oddeven_score * 0.20 +
        achrom_score * 0.10 +
        atmos_score * 0.10
    )
    return float(np.clip(composite, 0, 1))


def compute_ensemble_score(row: pd.Series) -> pd.Series:
    """Compute ensemble score, classification, confidence, and tier from a row."""
    physics_score = compute_physics_composite(row)
    ml_prob = row.get("ml_score", 0.5)
    
    # RESPECT STAGE 5 VETOES — do not override hard vetoes
    if row.get("vetoed", False):
        veto_cls = row.get("veto_cls", "FALSE_POSITIVE")
        return pd.Series({
            "ensemble_score": 0.05,
            "final_classification": veto_cls,
            "confidence": 0.05,
            "confidence_tier": "Low",
            "physics_composite": float(physics_score),
            "ml_score": float(ml_prob),
        })
    
    alias_type = row.get("harmonic_best_alias", None)
    alias_strength = row.get("harmonic_best_sde", 0.0)
    if isinstance(alias_strength, (np.ndarray, list, tuple)):
        alias_strength = float(alias_strength[0]) if len(alias_strength) > 0 else 0.0
    else:
        alias_strength = float(alias_strength) if pd.notna(alias_strength) else 0.0

    if physics_score < 0.2:
        final_score = physics_score * 0.8 + ml_prob * 0.2
        final_class = 'ECLIPSING_BINARY'
        confidence = 1 - final_score
        tier = 'Low'
    elif physics_score < 0.4 and ml_prob < 0.3:
        final_score = physics_score * 0.7 + ml_prob * 0.3
        final_class = 'FALSE_POSITIVE'
        confidence = 1 - final_score
        tier = 'Low'
    elif physics_score > 0.7 and ml_prob > 0.6:
        final_score = physics_score * PHYSICS_WEIGHT + ml_prob * ML_WEIGHT
        final_class = 'TRANSIT'
        confidence = final_score
        tier = 'High'
    elif physics_score > 0.5 and ml_prob > 0.4:
        final_score = physics_score * PHYSICS_WEIGHT + ml_prob * ML_WEIGHT
        final_class = 'TRANSIT'
        confidence = final_score
        tier = 'High' if final_score > 0.65 else 'Medium'
    elif physics_score > 0.5 or ml_prob > 0.5:
        final_score = physics_score * PHYSICS_WEIGHT + ml_prob * ML_WEIGHT
        final_class = 'UNCERTAIN'
        confidence = final_score
        tier = 'Medium'
    else:
        final_score = physics_score * PHYSICS_WEIGHT + ml_prob * ML_WEIGHT
        final_class = 'FALSE_POSITIVE'
        confidence = 1 - final_score
        tier = 'Low'
    
    if alias_type and alias_strength > 0.5:
        final_score *= 0.5
        if final_class == 'TRANSIT':
            final_class = 'UNCERTAIN'
            tier = 'Medium'
    
    return pd.Series({
        "ensemble_score": float(final_score),
        "final_classification": final_class,
        "confidence": float(confidence),
        "confidence_tier": tier,
        "physics_composite": float(physics_score),
        "ml_score": float(ml_prob),
    })

def run_ensemble_scoring(candidates: pd.DataFrame, config: dict) -> pd.DataFrame:
    logger.info("Stage 7: Ensemble scoring for %d candidates", len(candidates))

    if candidates is None or len(candidates) == 0:
        return pd.DataFrame()

    # Load physics features directly from Stage 4
    features_path = Path(config["paths"]["processed"]) / "features.csv"
    features_df = pd.read_csv(features_path) if features_path.exists() else pd.DataFrame()

    # Load Stage 5 ML results
    stage5_path = Path(config["paths"]["results"]) / "ml_predictions.csv"
    stage5_df = pd.read_csv(stage5_path) if stage5_path.exists() else pd.DataFrame()

    # Start with features as base (has all physics columns)
    if len(features_df) > 0:
        merged = features_df.copy()
    else:
        merged = candidates.copy()

    # Merge Stage 5 results
    if len(stage5_df) > 0:
        merge_cols = ["tic_id", "classification", "ml_confidence", "ml_score", "vetoed", "veto_cls"]
        merge_cols = [c for c in merge_cols if c in stage5_df.columns]
        merged = merged.merge(stage5_df[merge_cols], on="tic_id", how="left")

    # Fill missing ML columns
    for col in ["classification", "ml_confidence", "ml_score", "vetoed"]:
        if col not in merged.columns:
            if col == "classification":
                merged[col] = "UNCERTAIN"
            elif col == "vetoed":
                merged[col] = False
            else:
                merged[col] = 0.5

    # Add candidate info if missing
    if "period" not in merged.columns and len(candidates) > 0 and "period" in candidates.columns:
        cand_cols = ["tic_id", "period", "depth", "duration", "t0", "sde", "tls_sde"]
        cand_cols = [c for c in cand_cols if c in candidates.columns]
        merged = merged.merge(candidates[cand_cols], on="tic_id", how="left")

    # Apply ensemble scoring
    ensemble_results = merged.apply(compute_ensemble_score, axis=1)
    merged = pd.concat([merged.reset_index(drop=True), ensemble_results], axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    merged["noise_percent"] = ((1.0 - merged["ensemble_score"].clip(0, 1)) * 100).round(1)
    
    # Alias for Stage 8 compatibility
    merged["final_score"] = merged["ensemble_score"]

    print(f"\n{'='*100}")
    print(f"  DETAILED ENSEMBLE REPORT — {len(merged)} Candidates")
    print(f"  Physics Weight: {PHYSICS_WEIGHT*100:.0f}% | ML Weight: {ML_WEIGHT*100:.0f}%")
    print(f"{'='*100}\n")

    print(f"  {'TIC ID':<14} {'Final Class':<18} {'Overall':>8} {'Phys':>6} {'ML':>6} {'Tier':>8} {'Period':>8} {'Depth%':>8}")
    print(f"  {'-'*14} {'-'*18} {'-'*8} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")

    for _, row in merged.iterrows():
        cls = row["final_classification"]
        if cls == "TRANSIT":
            icon = "🪐"
        elif cls == "HARMONIC_ALIAS":
            icon = "〰️"
        elif cls == "UNCERTAIN":
            icon = "❓"
        elif cls in ("FALSE_POSITIVE", "ECLIPSING_BINARY", "BLEND", "STELLAR_ACTIVITY"):
            icon = "❌"
        else:
            icon = "⚪"

        period = row.get("period", row.get("period_x", row.get("period_y", 0)))
        depth = row.get("depth", row.get("depth_x", row.get("depth_y", 0)))

        print(f"  {icon} TIC {int(row['tic_id']):<10} {cls:<18} {row['ensemble_score']*100:>6.1f}% {row['physics_composite']*100:>5.1f}% {row['ml_score']*100:>5.1f}% {row['confidence_tier']:>8} {period:>8.3f} {depth*100:>8.3f}")

    print(f"\n{'='*100}")
    print(f"  CLASSIFICATION SUMMARY")
    print(f"{'='*100}")

    for cls, count in merged["final_classification"].value_counts().items():
        pct = count / len(merged) * 100
        bar = "█" * int(pct / 2)
        if cls == "TRANSIT":
            icon = "🪐"
        elif cls == "HARMONIC_ALIAS":
            icon = "〰️"
        elif cls == "UNCERTAIN":
            icon = "❓"
        elif cls in ("FALSE_POSITIVE", "ECLIPSING_BINARY", "BLEND", "STELLAR_ACTIVITY"):
            icon = "❌"
        else:
            icon = "⚪"
        print(f"  {icon} {cls:<20} {count:>2} ({pct:>5.1f}%) {bar}")

    print(f"\n  🎯 CONFIDENCE TIERS")
    print(f"  {'-'*20}")
    for tier, count in merged["confidence_tier"].value_counts().items():
        icon = "🟢" if tier == "High" else "🟡" if tier == "Medium" else "🔴"
        print(f"  {icon} {tier:<10} {count:>2} ({count/len(merged)*100:>5.1f}%)")

    print(f"\n  📊 TOTAL: {len(merged)} candidates analyzed")
    print(f"{'='*100}")

    out_path = Path(config["paths"]["results"]) / "candidate_predictions.csv"
    merged.to_csv(out_path, index=False)
    logger.info("Stage 7 complete: predictions saved to %s", out_path)

    transit_count = (merged["final_classification"] == "TRANSIT").sum()
    logger.info("  → %d TRANSIT candidates (High+Medium)", transit_count)

    return merged