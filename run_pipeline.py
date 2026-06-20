#!/usr/bin/env python3
"""Main pipeline runner — executes all 8 stages with detailed reporting."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tess_pipeline.config import ensure_dirs, load_config
from tess_pipeline.demo_data import generate_demo_dataset
from tess_pipeline.stage1_preprocess import run_preprocess
from tess_pipeline.stage2_detect import run_detection, detect_single
from tess_pipeline.stage3_gaia import run_gaia_crossmatch
from tess_pipeline.stage4_physics import run_physics_checks
from tess_pipeline.stage5_classify import run_ml_classification
from tess_pipeline.stage6_params import run_parameter_estimation
from tess_pipeline.stage7_ensemble import run_ensemble_scoring
from tess_pipeline.stage8_visualize import run_visualization
from tess_pipeline.utils import load_tic_ids, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TESS Transit Detection Pipeline — 8-stage analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo mode (synthetic data, no MAST needed):
  python3 run_pipeline.py --demo --demo-n 50

  # Full pipeline with TIC ID list:
  python3 run_pipeline.py --tic-ids data/tic_ids.csv --sector 14

  # Test Stage 2 on a single real TIC ID:
  python3 run_pipeline.py --test-stage2 1000003

  # Run specific stages:
  python3 run_pipeline.py --demo --stages 4,5,6,7,8
        """,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--tic-ids", type=str, default=None, help="CSV/text file of TIC IDs")
    parser.add_argument("--sector", type=int, default=None, help="TESS sector number")
    parser.add_argument("--max-curves", type=int, default=None, help="Limit number of curves")
    parser.add_argument("--skip-download", action="store_true", help="Skip MAST download (use cached)")
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo data")
    parser.add_argument("--demo-n", type=int, default=200, help="Number of demo curves")
    parser.add_argument(
        "--stages",
        type=str,
        default="1,2,3,4,5,6,7,8",
        help="Comma-separated stages to run (default: all)",
    )
    parser.add_argument(
        "--test-stage2",
        type=int,
        default=None,
        help="Test Stage 2 (BLS+TLS+Harmonic) on single TIC ID",
    )
    parser.add_argument(
        "--sde-threshold",
        type=float,
        default=None,
        help="Override BLS SDE threshold (default: 4.0 demo, 7.0 real)",
    )
    parser.add_argument(
        "--no-ai-features",
        action="store_true",
        help="Disable Harmonic analysis (run original pipeline only)",
    )
    parser.add_argument("--force-retrain", action="store_true", help="Force retrain ML model")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return parser.parse_args()


def run_stage2_test(tic_id: int, config: dict, args: argparse.Namespace) -> None:
    """Quick test of Stage 2 on a single TIC ID."""
    print(f"\n{'#'*70}")
    print(f"# STAGE 2 TEST MODE — TIC {tic_id}")
    print(f"# Features: BLS + TLS + Harmonic Analysis")
    print(f"{'#'*70}")
    
    sde = args.sde_threshold or 3.0
    
    result = detect_single(
        tic_id=tic_id,
        config=config,
        sde_threshold=sde,
        run_harmonic=True,
    )
    
    if result is None:
        print(f"\n  ❌ No detection for TIC {tic_id} at SDE={sde}")
        print(f"  Try: python3 run_pipeline.py --test-stage2 {tic_id} --sde-threshold 1.0")
        sys.exit(0)
    
    test_dir = Path("outputs/test_stage2")
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / f"tic_{tic_id}_stage2_test.json"
    import json
    with open(test_file, "w") as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in result.items() if not isinstance(v, dict)}, f, indent=2)
    
    print(f"\n{'#'*70}")
    print(f"# TEST COMPLETE — Result saved: {test_file}")
    print(f"{'#'*70}")
    sys.exit(0)


def run_pipeline(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_dirs(config)

    stages = [int(s.strip()) for s in args.stages.split(",")]
    sector = args.sector or config["pipeline"].get("sector")
    
    is_demo = args.demo
    config["_is_demo_mode"] = is_demo

    if args.test_stage2 is not None:
        run_stage2_test(args.test_stage2, config, args)

    # Resolve TIC IDs
    if args.demo:
        logger.info("=== DEMO MODE: generating %d synthetic light curves ===", args.demo_n)
        tic_ids, labels_df = generate_demo_dataset(config, n_curves=args.demo_n)
    elif args.tic_ids:
        tic_ids = load_tic_ids(args.tic_ids)
        labels_df = None
    elif config["pipeline"].get("tic_ids_file"):
        tic_ids = load_tic_ids(config["pipeline"]["tic_ids_file"])
        labels_df = None
    else:
        logger.error("No TIC IDs provided. Use --demo, --tic-ids, or set pipeline.tic_ids_file in config.")
        sys.exit(1)

    max_curves = args.max_curves or config["pipeline"].get("max_curves")
    if max_curves:
        tic_ids = tic_ids[: int(max_curves)]

    logger.info("Pipeline starting: %d targets, stages=%s", len(tic_ids), stages)

    candidates = None
    features = None
    classified = None
    predictions = None
    stage5_output = None
    
    if args.sde_threshold is not None:
        sde_threshold = args.sde_threshold
    elif args.demo:
        sde_threshold = 4.0
    else:
        sde_threshold = None

    run_harmonic = not args.no_ai_features

    if 1 in stages:
        logger.info("=" * 60)
        logger.info("STAGE 1: Preprocessing & Quality Control")
        logger.info("=" * 60)
        if not args.demo:
            run_preprocess(tic_ids, config, sector=sector, skip_download=args.skip_download)
        else:
            logger.info("Stage 1: Skipped (demo data already generated)")

    if 2 in stages:
        logger.info("=" * 60)
        logger.info("STAGE 2: BLS + TLS + Harmonic Detection")
        logger.info("=" * 60)
        
        candidates = run_detection(
            tic_ids, config,
            sde_threshold=sde_threshold,
            run_harmonic=run_harmonic,
        )

    if 3 in stages:
        logger.info("=" * 60)
        logger.info("STAGE 3: Gaia Cross-Match")
        logger.info("=" * 60)
        if candidates is None or len(candidates) == 0:
            logger.warning("No candidates from Stage 2 — skipping Gaia cross-match")
            candidates = pd.DataFrame()
        else:
            candidates = run_gaia_crossmatch(candidates, config)

    if 4 in stages:
        logger.info("=" * 60)
        logger.info("STAGE 4: Seven Physics Checks")
        logger.info("=" * 60)
        if candidates is None or len(candidates) == 0:
            logger.warning("No candidates from Stage 2/3 — skipping physics checks")
            features = pd.DataFrame()
        else:
            features = run_physics_checks(candidates, config)

    if 5 in stages:
        logger.info("=" * 60)
        logger.info("STAGE 5: ML Classification")
        logger.info("=" * 60)
        if features is None or len(features) == 0:
            logger.warning("No features from Stage 4 — skipping ML classification")
            classified = pd.DataFrame()
        else:
            classified = run_ml_classification(features, config, force_retrain=args.force_retrain)

    if 6 in stages:
        logger.info("=" * 60)
        logger.info("STAGE 6: Parameter Estimation")
        logger.info("=" * 60)
        if classified is None or len(classified) == 0:
            logger.warning("No classified candidates — skipping parameter estimation")
            stage5_output = pd.DataFrame()
        else:
            stage5_output = classified.copy()  # Save Stage 5 output before Stage 6 modifies it
            classified = run_parameter_estimation(classified, config)

    if 7 in stages:
        logger.info("=" * 60)
        logger.info("STAGE 7: Ensemble Scoring")
        logger.info("=" * 60)
        if stage5_output is None or len(stage5_output) == 0:
            logger.warning("No Stage 5 output — skipping ensemble scoring")
            predictions = pd.DataFrame()
        else:
            predictions = run_ensemble_scoring(stage5_output, config)

    if 8 in stages:
        logger.info("=" * 60)
        logger.info("STAGE 8: Visualization & Reports")
        logger.info("=" * 60)
        if predictions is None or len(predictions) == 0:
            logger.warning("No predictions from Stage 7 — skipping visualization")
            predictions = pd.DataFrame()
        else:
            run_visualization(predictions, config)

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL TERMINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  🚀 TESS TRANSIT PIPELINE — FINAL RESULTS")
    print("=" * 80)
    
    # Input summary
    print(f"\n  📥 INPUT SUMMARY")
    print(f"  {'-'*40}")
    print(f"  Total light curves processed: {len(tic_ids)}")
    if labels_df is not None and len(labels_df) > 0:
        print(f"  (Demo mode — synthetic labels available)")
        true_counts = labels_df["label"].value_counts()
        for label, count in true_counts.items():
            print(f"    └─ True {label}: {count}")
    else:
        print(f"  (Real data — no ground truth labels)")
    
    # Candidate summary
    pred_path = Path(config["paths"]["results"]) / "candidate_predictions.csv"
    if pred_path.exists():
        preds = pd.read_csv(pred_path)
        total = len(preds)
        
        print(f"\n  🔍 DETECTION SUMMARY")
        print(f"  {'-'*40}")
        print(f"  Candidates passing BLS/TLS: {total}")
        
        # Per-class breakdown with icons
        print(f"\n  📊 CLASSIFICATION BREAKDOWN")
        print(f"  {'-'*40}")
        
        class_icons = {
            "TRANSIT": "🪐", "ECLIPSE": "🌑", "BLEND": "🔀",
            "STELLAR_ACTIVITY": "☀️", "HARMONIC_ALIAS": "〰️",
            "UNCERTAIN": "❓", "REJECT": "❌", "NOISE": "📉",
            "ECLIPSING_BINARY": "🌑", "FALSE_POSITIVE": "❌",
        }
        
        cls_col = "final_classification" if "final_classification" in preds.columns else "classification"
        for cls in ["TRANSIT", "ECLIPSING_BINARY", "FALSE_POSITIVE", "BLEND", "STELLAR_ACTIVITY", "HARMONIC_ALIAS", "UNCERTAIN", "REJECT"]:
            count = (preds[cls_col] == cls).sum()
            if count > 0:
                pct = 100 * count / total
                bar = "█" * int(pct / 3)
                icon = class_icons.get(cls, "❓")
                print(f"  {icon} {cls:<20} {count:>3} ({pct:>5.1f}%) {bar}")
        
        # Confidence tiers
        print(f"\n  🎯 CONFIDENCE TIERS")
        print(f"  {'-'*40}")
        tier_col = "confidence_tier" if "confidence_tier" in preds.columns else "tier"
        for tier in ["High", "Medium", "Low"]:
            if tier_col in preds.columns:
                count = (preds[tier_col] == tier).sum()
            else:
                count = 0
            pct = 100 * count / total if total > 0 else 0
            icon = "🟢" if tier == "High" else "🟡" if tier == "Medium" else "🔴"
            print(f"  {icon} {tier:<10} {count:>3} ({pct:>5.1f}%)")
        
        # Top candidates table
        print(f"\n  🏆 TOP CANDIDATES")
        print(f"  {'-'*40}")
        score_col = "ensemble_score" if "ensemble_score" in preds.columns else "final_score" if "final_score" in preds.columns else "ml_confidence"
        top = preds[preds[cls_col].str.startswith("TRANSIT", na=False)].nlargest(5, score_col)
        if len(top) > 0:
            print(f"  {'TIC ID':<10} {'Score':>6} {'Period(d)':>10} {'Depth(%)':>10} {'Conf':>8} {'Tier':>8}")
            print(f"  {'-'*10} {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
            for _, row in top.iterrows():
                depth = row["depth"]
                depth_pct = depth * 100
                conf = row.get("ensemble_score", row.get("final_score", row.get("ml_confidence", 0.5)))
                tier = row.get("confidence_tier", row.get("tier", "Medium"))
                print(f"  {int(row['tic_id']):<10} {conf:>6.2f} {row['period']:>10.3f} {depth_pct:>10.3f} {conf:>8.2f} {tier:>8}")
        else:
            print("  No transit candidates found.")
        
        # Physics checks
        print(f"\n  🔬 PHYSICS CHECKS (average scores)")
        print(f"  {'-'*40}")
        for check, col in [("Depth", "depth_score"), ("Shape", "shape_score"), ("NoSec", "nosec_score"), 
                          ("OddEven", "oddeven_score"), ("Achrom", "achrom_score"), ("Atmos", "atmos_score")]:
            if col in preds.columns:
                score = preds[col].mean()
                bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                print(f"  {check:<12} {score:>5.2f}  [{bar}]")
    
    print(f"\n  📁 OUTPUT FILES")
    print(f"  {'-'*40}")
    print(f"  Predictions:  {config['paths']['results']}/candidate_predictions.csv")
    print(f"  Parameters:   {config['paths']['results']}/candidate_parameters.csv")
    print(f"  Plots:        {config['paths']['plots']}/")
    print(f"  Report:       {config['paths']['reports']}/pipeline_report.pdf")
    print("=" * 80 + "\n")
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


def main() -> None:
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    run_pipeline(args)


if __name__ == "__main__":
    main()