#!/usr/bin/env python3
"""Run pipeline with patches applied — ZERO changes to core pipeline."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Apply patches FIRST (before importing any pipeline modules)
from patches import apply_all_patches
apply_all_patches()

# Now import and run the ORIGINAL run_pipeline.py from root
import run_pipeline

# ============================================================
# CRITICAL FIX: Monkey-patch stage7_ensemble AFTER all imports
# This overrides whatever the patches module set
# ============================================================
import tess_pipeline.stage7_ensemble as s7

s7.ML_WEIGHT = 0.15
s7.PHYSICS_WEIGHT = 0.55
s7.DETECTION_WEIGHT = 0.30
s7.MIN_PLANET_SCORE = 0.35
s7.MIN_UNCERTAIN_SCORE = 0.20
s7.ML_BYPASS_PHYSICS_THRESHOLD = 0.50
s7.ML_BYPASS_BLS_SDE = 8.0
s7.ML_BYPASS_TLS_SDE = 12.0
s7.VSHAPE_TOLERANCE = 0.4
s7.ODDEVEN_TOLERANCE = 0.3

# Also monkey-patch the functions if they were already imported elsewhere
import tess_pipeline.stage7_ensemble
tess_pipeline.stage7_ensemble.ML_WEIGHT = 0.15
tess_pipeline.stage7_ensemble.PHYSICS_WEIGHT = 0.55
tess_pipeline.stage7_ensemble.DETECTION_WEIGHT = 0.30

print("\n" + "="*70)
print("  🔥 ENSEMBLE FIX vFINAL INJECTED")
print("  ML: 15% | Physics: 55% | Detection: 30%")
print("  ML BYPASS: Physics≥0.5 + BLS≥8 + TLS≥12 → ML floor=0.30")
print("="*70 + "\n")

if __name__ == "__main__":
    run_pipeline.main()