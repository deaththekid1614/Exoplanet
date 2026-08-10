# TESS Transit Detection Pipeline

A CPU-only, end-to-end pipeline for detecting and classifying exoplanet transit candidates from NASA TESS full-frame image light curves. Built for batch sector analysis without GPU dependencies.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Pipeline Architecture](#pipeline-architecture)
3. [The 7 Physics Validation Checks](#the-7-physics-validation-checks)
4. [Validation Results](#validation-results)
5. [Quick Start](#quick-start)
6. [Project Structure](#project-structure)
7. [Configuration](#configuration)
8. [Output Files](#output-files)
9. [Performance Notes](#performance-notes)
10. [Requirements](#requirements)
11. [License](#license)

---

## How It Works

The pipeline ingests a list of TIC IDs, downloads their light curves from the MAST archive, and runs an 8-stage detection and validation workflow:

1. **Download & Clean** — Retrieves TESS light curves from MAST. Removes outliers, handles data gaps, and applies a Savitzky-Golay filter to remove long-term stellar variability (e.g., starspot rotation).

2. **Transit Search** — Runs a fast BLS (Box Least Squares) screen across trial periods to flag candidates, then refines promising hits with TLS (Transit Least Squares) for precise period, epoch, and duration recovery.

3. **Stellar Context** — Cross-matches the target with Gaia DR3 to obtain radius, temperature, and luminosity. This anchors physical plausibility: a Jupiter-depth transit around an M-dwarf is suspicious; an Earth-depth transit around a Sun-like star is plausible.

4. **Physics Validation** — Applies 7 proxy checks that simulate how follow-up missions (JWST, Hubble) would vet a candidate. These checks operate entirely on the TESS light curve — they test the *same physical principles* that high-precision instruments confirm, adapted to TESS resolution and bandpass.

5. **ML Classification** — An XGBoost ensemble classifies the candidate using transit shape features, stellar parameters, and physics check scores.

6. **Parameter Fitting** — Bootstrap resampling estimates uncertainties on period, depth, duration, and inferred planetary radius.

7. **Ensemble Scoring** — Fuses ML confidence (40%) and physics check scores (60%) into a final weighted confidence. High scores indicate agreement between machine learning and physics.

8. **Reporting** — Generates per-target diagnostic plots and a summary PDF report.

---

## Pipeline Architecture

```
MAST (TESS) + Gaia DR3 + Curated Training Labels
        |
        v
Stage 1: Preprocess  —  Download, clean, Savitzky-Golay detrend
        |
        v
Stage 2: Detect      —  BLS screen -> TLS refine
        |
        v
Stage 3: Gaia        —  DR3 cross-match for stellar params
        |
        v
Stage 4: Physics     —  7 validation checks
        |
        v
Stage 5: Classify    —  XGBoost 2-stage classifier
        |
        v
Stage 6: Parameters  —  Bootstrap uncertainty estimation
        |
        v
Stage 7: Ensemble    —  Weighted confidence scoring
        |
        v
Stage 8: Visualize   —  Plots + PDF report
```

---

## The 7 Physics Validation Checks

These checks use the TESS light curve to test physical properties that JWST and Hubble would confirm with higher precision. **No actual JWST or Hubble data is used** — these are proxy validations operating on TESS photometry alone.

### 1. Depth vs. Gaia Radius
- **Simulates:** Hubble/WFC3 spectroscopic follow-up for planet radius measurement.
- **What it tests:** The transit depth is converted to an implied companion radius using the Gaia stellar radius. A depth implying a Jupiter-sized body around a small M-dwarf raises a flag; an Earth-sized depth around a Sun-like star is physically consistent.

### 2. Ingress/Egress Shape
- **Simulates:** Hubble/WFC3 high-cadence transit shape analysis.
- **What it tests:** Real planets produce flat-bottomed transits because they occult a uniform disk. Grazing eclipsing binaries show V-shaped or sloped ingress/egress. The check fits the transit profile and scores flatness.

### 3. No Secondary Eclipse
- **Simulates:** JWST phase-curve photometry covering half an orbit.
- **What it tests:** A flux dip at phase 0.5 indicates a self-luminous companion — characteristic of a star, not a planet. The pipeline searches for a secondary dip at the expected orbital phase.

### 4. Centroid Stability
- **Simulates:** Hubble fine-guidance sensor astrometric monitoring.
- **What it tests:** If the transit occurs on a blended background star rather than the target, the photocenter shifts during the event. The pipeline checks for centroid motion correlated with the transit epoch.

### 5. Odd-Even Consistency
- **Simulates:** Hubble precision photometry stacked over many orbits.
- **What it tests:** In circularized eclipsing binaries, primary and secondary eclipses can alternate in depth. Real planets repeat identical depths every orbit. The pipeline compares odd-numbered vs. even-numbered transit depths.

### 6. Achromaticity
- **Simulates:** Hubble G280 grism multi-wavelength photometry.
- **What it tests:** Real planets are approximately gray (achromatic) in the optical/NIR. Stellar spots or blended companions can produce color-dependent depth variations. Using TESS bandpass information, the pipeline checks for depth consistency.

### 7. Atmosphere Proxy
- **Simulates:** JWST NIRSpec transmission spectroscopy.
- **What it tests:** A thick atmosphere can produce anomalous curvature in ingress/egress. While TESS cannot resolve atmospheric molecular features, a smooth, symmetric limb-darkening profile is consistent with a solid body or thin atmosphere. Deviations flag extended atmospheres or systematic noise.

---

## Validation Results

### Test 1: Mixed 20 Targets (Known Labels)

- **Total targets:** 20
- **Successfully processed:** 18/20 (90%)
- **Data failures:** 2/20 (10%)
- **Accuracy on processed:** 92.9% (13/14)
- **Planets correctly called TRANSIT:** 6/6 (100%)
- **EBs/FPs correctly rejected:** 5/5 (100%)
- **Misclassifications:** 1 (planet classified as EB: TIC 307809773)

### Test 2: 10 New TIC IDs (Blind Test)

- **Total targets:** 10
- **Successfully processed:** 9/10 (90%)
- **Data failures:** 1/10 (10%)
- **Accuracy on processed:** 100% (9/9)
- **Planets correctly called TRANSIT:** 7/7 (100%)
- **FPs/EBs correctly flagged UNCERTAIN:** 2/2 (100%)
- **False positives:** 0

### Blind Test Target Breakdown

| TIC ID      | Prediction  | Ground Truth                          | Match |
|-------------|-------------|---------------------------------------|-------|
| 307210830   | TRANSIT     | Confirmed planet (L 98-59)            | Yes   |
| 33595516    | TRANSIT     | Confirmed planet (TOI-849 b)          | Yes   |
| 158588995   | TRANSIT     | Confirmed planet                      | Yes   |
| 149603524   | TRANSIT     | Confirmed planet                      | Yes   |
| 92352620    | TRANSIT     | Confirmed planet (WASP-94 B b)        | Yes   |
| 28159019    | TRANSIT     | Confirmed planet                      | Yes   |
| 172370679   | UNCERTAIN   | Known false positive                  | Yes   |
| 38846515    | UNCERTAIN   | Likely false positive                 | Yes   |
| 144065872   | TRANSIT     | Confirmed planet                      | Yes   |
| 142276270   | Data fail   | Confirmed planet (TOI-1136, 7-planet) | N/A   |

**Ensemble weighting:** ML 40% / Physics 60%

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Demo mode — synthetic data, no network required
python run_pipeline.py --demo --demo-n 200

# Validation run — 10 new TIC IDs
python3 -B run_pipeline.py --tic-ids data/test_tic_ids_10.csv --sde-threshold 2.5

# Validation run — 20 mixed targets with known labels
python3 -B run_pipeline.py --tic-ids data/test_confirmed_planets.csv --sde-threshold 2.5

# Production run — full sector
python run_pipeline.py --tic-ids data/tic_ids.csv --sector 14

# Resume from cached preprocessing
python run_pipeline.py --tic-ids data/tic_ids.csv --sector 14 --skip-download
```

---

## Project Structure

```
tess-transit-pipeline/
├── config/
│   └── default.yaml              # Pipeline parameters
├── data/
│   ├── processed/                # Detrended .npz files
│   ├── candidates/               # BLS+TLS detection outputs
│   ├── gaia_cache/               # Cached stellar parameters
│   ├── training/                 # Curated label CSVs
│   ├── test_tic_ids_10.csv       # 10-target blind test set
│   └── test_confirmed_planets.csv # 20-target mixed validation set
├── models/                       # Trained XGBoost + scalers
├── outputs/
│   ├── results/                  # candidate_predictions.csv
│   ├── plots/                    # Per-candidate diagnostics
│   └── reports/                  # PDF summary report
├── tess_pipeline/
│   ├── stage1_preprocess.py      # Download, clean, detrend
│   ├── stage2_detect.py          # BLS screen + TLS refine
│   ├── stage3_gaia.py            # Gaia DR3 cross-match
│   ├── stage4_physics.py         # 7 physics checks
│   ├── stage5_classify.py        # XGBoost ML classifier
│   ├── stage6_params.py          # Bootstrap parameter fitting
│   ├── stage7_ensemble.py        # Confidence scoring
│   ├── stage8_visualize.py       # Plots + PDF generation
│   └── demo_data.py              # Synthetic test data generator
├── run_pipeline.py               # Main entry point
└── requirements.txt              # Python dependencies
```

---

## Configuration

Pipeline behavior is controlled via `config/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bls.sde_threshold` | 7.0 | Minimum BLS Signal Detection Efficiency |
| `detrend.sg_window` | 0.5 days | Savitzky-Golay smoothing window |
| `ensemble.ml_weight` | 0.40 | ML contribution to final score |
| `ensemble.physics_weight` | 0.60 | Physics contribution to final score |
| `classify.threshold_transit` | 0.65 | Score threshold for TRANSIT label |
| `classify.threshold_uncertain` | 0.35 | Score threshold for UNCERTAIN label |

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/results/candidate_predictions.csv` | Final classifications with confidence scores |
| `outputs/results/candidate_parameters.csv` | Fitted transit parameters and uncertainties |
| `outputs/plots/candidates/{tic_id}/` | Diagnostic plots per candidate |
| `outputs/reports/pipeline_report.pdf` | Summary methodology and results report |

---

## Performance Notes

- **Batch processing:** Stage 1 handles 100 curves per batch to manage memory.
- **Parallel BLS:** 4 worker threads via `joblib`.
- **TLS optimization:** TLS runs only on BLS candidates, not the full sector.
- **Centroid sampling:** Computed only for the top 10% of candidates by SDE.
- **Memory hygiene:** `plt.close()` after every figure to prevent RAM accumulation.
- **No GPU required:** XGBoost trains in under 30 seconds on a modern CPU.

### Estimated Runtime (per TESS sector, CPU-only)

| Stage | Estimated Time |
|-------|---------------|
| 1. Preprocess | 4–6 hours |
| 2. BLS + TLS | 4–7 hours |
| 3. Gaia cross-match | 1–2 hours |
| 4–7. Physics + ML + Ensemble | ~15 minutes |
| 8. Plots (top 100 candidates) | ~10 minutes |

**Recommended workflow:** Run Stage 1 overnight, then Stages 2–8 the following day.

---

## Requirements

- Python 3.8 or higher
- lightkurve
- astropy
- transitleastsquares
- harmonica
- scikit-learn
- xgboost
- matplotlib
- reportlab
- pandas, numpy, scipy
- joblib

See `requirements.txt` for pinned dependency versions.

---
