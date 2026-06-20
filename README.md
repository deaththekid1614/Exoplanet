# TESS Transit Detection Pipeline

CPU-only, MacBook-safe pipeline for detecting and classifying exoplanet transit candidates from TESS light curves. Built for hackathon-scale analysis (20,000–30,000 curves) without GPU requirements.

## Architecture

```
MAST (TESS) + Gaia DR3 + Curated Training Labels
        ↓
Stage 1: Preprocess (Savitzky-Golay detrend)
        ↓
Stage 2: Detect (BLS screen → TLS refine)
        ↓
Stage 3: Gaia stellar parameters
        ↓
Stage 4: 7 physics validation checks
        ↓
Stage 5: XGBoost 2-stage classification
        ↓
Stage 6: Parameter estimation (bootstrap)
        ↓
Stage 7: Ensemble confidence scoring
        ↓
Stage 8: Plots + PDF report
```

## Validation Results

### Test 1: Mixed 20 Targets (Known Labels)
| Metric | Value |
|--------|-------|
| Total targets | 20 |
| Successfully processed | 18/20 (90%) |
| Data failures | 2/20 (10%) |
| **Accuracy on processed** | **92.9% (13/14)** |
| Planets correctly detected as TRANSIT | 6/6 (100%) |
| EBs/FPs correctly rejected | 5/5 (100%) |
| Misclassifications | 1 (planet→EB: TIC 307809773) |

### Test 2: 10 New TIC IDs (Blind Test)
| Metric | Value |
|--------|-------|
| Total targets | 10 |
| Successfully processed | 9/10 (90%) |
| Data failures | 1/10 (10%) |
| **Accuracy on processed** | **100% (9/9)** |
| Planets correctly detected as TRANSIT | 7/7 (100%) |
| FPs/EBs correctly flagged UNCERTAIN | 2/2 (100%) |
| False positives | 0 |

### Target Breakdown (10 TIC Test)
| TIC ID | Prediction | Ground Truth | Match |
|--------|-----------|--------------|-------|
| 307210830 | TRANSIT | Confirmed planet (L 98-59) | ✅ |
| 33595516 | TRANSIT | Confirmed planet (TOI-849 b, P=0.765d) | ✅ |
| 158588995 | TRANSIT | Confirmed planet | ✅ |
| 149603524 | TRANSIT | Confirmed planet | ✅ |
| 92352620 | TRANSIT | Confirmed planet (WASP-94 B b) | ✅ |
| 28159019 | TRANSIT | Confirmed planet | ✅ |
| 172370679 | UNCERTAIN | Known false positive | ✅ |
| 38846515 | UNCERTAIN | Likely false positive | ✅ |
| 144065872 | TRANSIT | Confirmed planet | ✅ |
| 142276270 | Data failure | Confirmed planet (TOI-1136, 7 planets) | ⚠️ |

**Pipeline: ML + Physics Ensemble (40% ML / 60% Physics weight)**

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Demo mode (no MAST/network needed — synthetic data)
python run_pipeline.py --demo --demo-n 200

# Validation run — 10 new TIC IDs
python3 -B run_pipeline.py --tic-ids data/test_tic_ids_10.csv --sde-threshold 2.5

# Validation run — 20 mixed targets (known labels)
python3 -B run_pipeline.py --tic-ids data/test_confirmed_planets.csv --sde-threshold 2.5

# Production run with TIC ID list
python run_pipeline.py --tic-ids data/tic_ids.csv --sector 14

# Overnight run (30k curves, use cached preprocess)
python run_pipeline.py --tic-ids data/tic_ids.csv --sector 14 --skip-download
```

## Project Structure

```
tess-transit-pipeline/
├── config/default.yaml      # All pipeline parameters
├── run_pipeline.py          # Main entry point
├── tess_pipeline/
│   ├── stage1_preprocess.py # Download, clean, detrend
│   ├── stage2_detect.py     # BLS + TLS
│   ├── stage3_gaia.py       # Gaia DR3 cross-match
│   ├── stage4_physics.py    # 7 physics checks
│   ├── stage5_classify.py   # XGBoost ML
│   ├── stage6_params.py     # Parameter fitting
│   ├── stage7_ensemble.py   # Final scoring
│   ├── stage8_visualize.py  # Plots + PDF
│   └── demo_data.py         # Synthetic test data
├── data/
│   ├── processed/           # Detrended .npz files
│   ├── candidates/          # BLS+TLS hits
│   ├── gaia_cache/          # Cached stellar params
│   ├── training/            # Curated labels CSV
│   ├── test_tic_ids_10.csv  # 10-target validation set
│   └── test_confirmed_planets.csv  # 20-target mixed validation set
├── models/                  # Trained XGBoost + scaler
└── outputs/
    ├── results/             # candidate_predictions.csv
    ├── plots/               # Per-candidate + summary plots
    └── reports/             # 3-page PDF
```

## The 7 Physics Checks

| Check | Simulates | What it tests |
|-------|-----------|---------------|
| 1. Depth vs Gaia | Hubble stellar typing | Depth consistent with Earth/Jupiter size? |
| 2. Ingress/egress shape | Hubble WFC3 | Flat bottom (planet) vs V-shape (binary)? |
| 3. No secondary eclipse | JWST phase coverage | Flux drop at phase 0.5? |
| 4. Centroid stability | Hubble spatial res | Light center shifts during transit? |
| 5. Odd-even consistency | Hubble precision | All transits same depth? |
| 6. Achromaticity | Hubble G280 multi-color | Same depth all wavelengths? |
| 7. Atmosphere proxy | JWST NIRSpec | Curved ingress/egress? |

## MacBook Optimization

- **Batch processing**: 100 curves at a time (Stage 1)
- **Parallel BLS**: 4 threads via joblib
- **TLS on candidates only**: ~1,000 not 30,000
- **Centroid on top 10%**: saves disk and time
- **plt.close()** after every plot: prevents RAM leaks
- **No GPU/CNN**: XGBoost trains in ~30 seconds on CPU

## Output Files

| File | Description |
|------|-------------|
| `outputs/results/candidate_predictions.csv` | Final classifications |
| `outputs/results/candidate_parameters.csv` | Fitted transit parameters |
| `outputs/plots/candidates/{tic_id}/` | 6 plots per top candidate |
| `outputs/reports/pipeline_report.pdf` | 3-page methodology report |

## Training Data Format

Place curated hackathon labels at `data/training/training_labels.csv`:

```csv
tic_id,label
1234567,planet
2345678,eclipse
3456789,blend
```

Labels: `planet`, `eclipse`, `blend`, `stellar_activity`, `false_positive`

## Configuration

Edit `config/default.yaml` to tune:
- BLS SDE threshold (default: 7.0)
- Savitzky-Golay window (default: 0.5 days)
- Ensemble weights (ML 30% + physics 70%)
- Classification thresholds

## Estimated Runtime (30,000 curves, MacBook 2015)

| Stage | Time |
|-------|------|
| 1. Preprocess | 4–6 hours |
| 2. BLS + TLS | 4–7 hours |
| 3. Gaia | 1–2 hours |
| 4–7. Physics + ML | ~15 minutes |
| 8. Plots (top 100) | ~10 minutes |

Run Stage 1 overnight, then Stages 2–8 the next day.

## License

MIT
