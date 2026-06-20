"""Stage 2: BLS fast screen + TLS refinement + Harmonic Analysis — FIXED."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.timeseries import BoxLeastSquares
from tqdm import tqdm

from tess_pipeline.stage1_preprocess import load_processed_curve
from tess_pipeline.harmonic.harmonic_analysis import analyze_harmonics

logger = logging.getLogger(__name__)

u_day = u.day

# ─── CONFIG ─────────────────────────────────────────────────────
SDE_PRIMARY = 4.0
SDE_SECONDARY = 2.5
DURATION_GRID = np.linspace(0.03, 0.15, 20)
PERIOD_MIN = 0.5
PERIOD_MAX = 30.0
PERIOD_GRID = 10000


# =============================================================================
# BLS DETECTION — FIXED (accepts sde_threshold parameter)
# =============================================================================
def run_bls(time, flux, flux_err, tic_id, sde_threshold=None, **kwargs):
    """Run BLS with primary threshold, fallback to secondary."""
    primary = sde_threshold if sde_threshold is not None else SDE_PRIMARY
    secondary = primary * 0.625 if sde_threshold is not None else SDE_SECONDARY

    model = BoxLeastSquares(time, flux, dy=flux_err)
    periods = np.linspace(PERIOD_MIN, PERIOD_MAX, PERIOD_GRID)

    results = model.power(periods, DURATION_GRID)
    best_idx = np.argmax(results.power)
    best_power = results.power[best_idx]
    best_period = results.period[best_idx]
    best_duration = results.duration[best_idx]
    best_t0 = results.transit_time[best_idx]

    power_mean = np.mean(results.power)
    power_std = np.std(results.power)
    sde = (best_power - power_mean) / power_std if power_std > 0 else 0

    depth = results.depth[best_idx] if hasattr(results, 'depth') else 0.001

    if sde >= primary:
        return {
            'period': float(best_period.value) if hasattr(best_period, 'value') else float(best_period),
            'epoch': float(best_t0.value) if hasattr(best_t0, 'value') else float(best_t0),
            'duration': float(best_duration.value) if hasattr(best_duration, 'value') else float(best_duration),
            'depth': float(depth),
            'sde': float(sde),
            'power': float(best_power),
            'pass': 'primary'
        }

    if sde >= secondary:
        folded = (time - best_t0) % best_period
        in_transit = np.abs(folded - best_period/2) < best_duration/2
        if np.sum(in_transit) > 10:
            transit_depths = []
            for epoch in range(int((time.max() - time.min()) / best_period) + 1):
                t_center = best_t0 + epoch * best_period
                mask = np.abs(time - t_center) < best_duration / 2
                if np.sum(mask) > 2:
                    transit_depths.append(np.median(flux[mask]))
            if len(transit_depths) > 1:
                depth_consistency = np.std(transit_depths) / (np.mean(transit_depths) + 1e-10)
                if depth_consistency < 0.5:
                    return {
                        'period': float(best_period.value) if hasattr(best_period, 'value') else float(best_period),
                        'epoch': float(best_t0.value) if hasattr(best_t0, 'value') else float(best_t0),
                        'duration': float(best_duration.value) if hasattr(best_duration, 'value') else float(best_duration),
                        'depth': float(depth),
                        'sde': float(sde),
                        'power': float(best_power),
                        'pass': 'secondary'
                    }

    return None


def _safe_float(val):
    if val is None:
        return 0.0
    if isinstance(val, (np.ndarray, list, tuple)) and len(val) > 0:
        return float(val[0])
    return float(val)


def run_tls(time: np.ndarray, flux: np.ndarray, bls_hit: dict, config: dict) -> dict:
    import transitleastsquares as tls

    flux_err = np.full_like(flux, np.std(flux) * 0.001 + 1e-6)
    model = tls.transitleastsquares(time, flux, flux_err)

    try:
        result = model.power(
            period_min=bls_hit["period"] * 0.95,
            period_max=bls_hit["period"] * 1.05,
        )
    except Exception:
        return bls_hit

    if result is None:
        return bls_hit

    period_val = getattr(result, 'period', None)
    if period_val is None:
        return bls_hit

    try:
        period = _safe_float(period_val)
        duration = _safe_float(getattr(result, 'duration', bls_hit.get('duration', 0)))
        transit_time = _safe_float(getattr(result, 'transit_time', bls_hit.get('epoch', 0)))
        sde = _safe_float(getattr(result, 'SDE', 0))

        tls_depth_raw = _safe_float(getattr(result, 'depth', bls_hit.get('depth', 0)))

        if tls_depth_raw > 0.5:
            from tess_pipeline.utils import phase_fold
            phase, folded = phase_fold(time, flux, period, transit_time)
            half_dur = (duration / period) / 2.0
            in_transit = (phase < half_dur) | (phase > 1 - half_dur)
            oot = (phase > 0.1) & (phase < 0.9)

            if np.sum(in_transit) > 0 and np.sum(oot) > 0:
                in_median = np.median(folded[in_transit])
                oot_median = np.median(folded[oot])
                depth = max(0.0, oot_median - in_median)
            else:
                depth = bls_hit.get('depth', 0.001)
        else:
            depth = tls_depth_raw

    except (TypeError, ValueError) as e:
        logger.debug("TLS result parsing failed: %s", e)
        return bls_hit

    return {
        **bls_hit,
        "period": period,
        "duration": duration,
        "depth": depth,
        "t0": transit_time,
        "tls_sde": sde,
    }


def detect_single(tic_id: int, config: dict, run_tls_refinement: bool = True,
                  sde_threshold: float | None = None,
                  run_harmonic: bool = True) -> dict | None:
    out_buf = io.StringIO()

    def buf_print(*args, **kwargs):
        print(*args, file=out_buf, **kwargs)

    buf_print(f"\n{'='*70}")
    buf_print(f"  TIC {tic_id} — Full Detection & Analysis")
    buf_print(f"{'='*70}")

    data = load_processed_curve(tic_id, config)
    if data is None:
        buf_print(f"  ❌ No processed data found for TIC {tic_id}")
        buf_print(f"{'='*70}\n")
        print(out_buf.getvalue(), end="")
        return None

    time = data["time"]
    flux = data.get("detrended", data["flux"])
    flux_err = data.get("flux_err", np.ones_like(flux) * 0.001)
    buf_print(f"  📊 Data points: {len(time):,}")

    buf_print(f"\n  [1/4] BLS Fast Screen")
    bls_hit = run_bls(time, flux, flux_err, tic_id, sde_threshold=sde_threshold)
    if bls_hit is None:
        buf_print(f"  ❌ No BLS detection above threshold")
        buf_print(f"{'='*70}\n")
        print(out_buf.getvalue(), end="")
        return None

    buf_print(f"    ✅ BLS HIT — Period: {bls_hit['period']:.4f} d | "
              f"Depth: {bls_hit['depth']:.6f} | SDE: {bls_hit['sde']:.2f}")

    if run_tls_refinement:
        buf_print(f"\n  [2/4] TLS Refinement")
        bls_hit = run_tls(time, flux, bls_hit, config)
        buf_print(f"    ✅ TLS — Period: {bls_hit['period']:.4f} d | "
                  f"Duration: {bls_hit['duration']:.4f} d | "
                  f"TLS SDE: {bls_hit.get('tls_sde', 0):.2f}")

    if run_harmonic:
        buf_print(f"\n  [3/4] Harmonic Analysis")
        try:
            harmonic_result = analyze_harmonics(
                time, flux,
                bls_hit["period"],
                bls_hit["t0"],
                bls_hit["duration"],
                bls_hit["depth"]
            )
            bls_hit.update(harmonic_result)

            h_flag = bool(bls_hit.get("harmonic_flag", False))
            h_alias = bls_hit.get("harmonic_best_alias")
            h_sde = bls_hit.get("harmonic_best_sde", 0.0)

            if isinstance(h_alias, (np.ndarray, list, tuple)):
                h_alias = h_alias[0] if len(h_alias) > 0 else None
            if isinstance(h_sde, (np.ndarray, list, tuple)):
                h_sde = float(h_sde[0]) if len(h_sde) > 0 else 0.0
            else:
                h_sde = float(h_sde)

            if harmonic_result["recommendation"] == "REJECT":
                buf_print(f"    ❌ SUB-HARMONIC ALIAS — likely false period")
            elif harmonic_result["recommendation"] == "REVIEW":
                buf_print(f"    ⚠️  SUB-HARMONIC ALIAS detected — review needed")
            else:
                buf_print(f"    ✅ Primary period genuine")
                if harmonic_result["harmonic_flags"]:
                    buf_print(f"       Notes: {', '.join(harmonic_result['harmonic_flags'])}")
        except Exception as e:
            buf_print(f"    ⚠️  Harmonic analysis failed: {e}")
            bls_hit["harmonic_flag"] = False
            bls_hit["harmonic_best_alias"] = None
            bls_hit["harmonic_best_sde"] = 0.0

    buf_print(f"\n  [4/4] Detection Complete — Physics checks in Stage 4")

    bls_hit["tic_id"] = tic_id

    period_v = float(bls_hit.get('period', 0))
    duration_v = float(bls_hit.get('duration', 0))
    depth_v = float(bls_hit.get('depth', 0))
    t0_v = float(bls_hit.get('t0', 0))
    bls_sde_v = float(bls_hit.get('sde', 0))
    tls_sde_raw = bls_hit.get('tls_sde', 'N/A')
    tls_sde_v = float(tls_sde_raw) if tls_sde_raw != 'N/A' and tls_sde_raw is not None else None

    h_flag_v = bool(bls_hit.get("harmonic_flag", False))

    buf_print(f"\n{'='*70}")
    buf_print(f"  TIC {tic_id} — Detection Summary")
    buf_print(f"{'='*70}")
    buf_print(f"  Period:        {period_v:.4f} d")
    buf_print(f"  Duration:      {duration_v:.4f} d")
    buf_print(f"  Depth:         {depth_v:.6f}")
    buf_print(f"  Epoch (T0):    {t0_v:.4f}")
    buf_print(f"  BLS SDE:       {bls_sde_v:.2f}")
    if tls_sde_v is not None:
        buf_print(f"  TLS SDE:       {tls_sde_v:.2f}")
    else:
        buf_print(f"  TLS SDE:       N/A")
    buf_print(f"  Harmonic Flag: {h_flag_v}")
    buf_print(f"{'='*70}\n")

    print(out_buf.getvalue(), end="")
    out_buf.close()

    return bls_hit


def run_bls_screen(tic_ids: list[int], config: dict,
                   sde_threshold: float | None = None,
                   run_harmonic: bool = True) -> pd.DataFrame:
    threshold = sde_threshold if sde_threshold is not None else config["detection"]["bls_sde_threshold"]

    print(f"\n{'#'*70}")
    print(f"# STAGE 2A: BLS Fast Screen + Harmonic Analysis")
    print(f"# Targets: {len(tic_ids)} | SDE Threshold: {threshold:.1f}")
    print(f"{'#'*70}\n")

    logger.info("Stage 2A: BLS screen on %d curves (SDE threshold: %.1f)", len(tic_ids), threshold)

    results = []
    for tic_id in tqdm(tic_ids, desc="BLS+Harmonic"):
        try:
            result = detect_single(
                tic_id, config,
                run_tls_refinement=True,
                sde_threshold=threshold,
                run_harmonic=run_harmonic
            )
            results.append(result)
        except Exception as e:
            logger.warning("Detection failed for TIC %s: %s", tic_id, e)
            results.append(None)

    hits = [r for r in results if r is not None]
    df = pd.DataFrame(hits)

    print(f"\n{'#'*70}")
    print(f"# STAGE 2A COMPLETE")
    print(f"# Candidates: {len(df)} / {len(tic_ids)} ({100*len(df)/len(tic_ids) if tic_ids else 0:.1f}%)")
    print(f"{'#'*70}\n")

    logger.info("Stage 2A: %d BLS candidates (%.1f%%)", len(df), 100 * len(df) / len(tic_ids) if tic_ids else 0)
    return df


def run_tls_refinement(candidates: pd.DataFrame, config: dict) -> pd.DataFrame:
    if len(candidates) == 0:
        return candidates

    print(f"\n{'#'*70}")
    print(f"# STAGE 2B: TLS Refinement (already done inline)")
    print(f"# Candidates: {len(candidates)}")
    print(f"{'#'*70}\n")

    out_path = Path(config["paths"]["candidates"]) / "candidates.csv"
    candidates.to_csv(out_path, index=False)

    print(f"\n{'#'*70}")
    print(f"# STAGE 2B COMPLETE")
    print(f"# Refined: {len(candidates)} | Saved: {out_path}")
    print(f"{'#'*70}\n")

    logger.info("Stage 2B: %d refined candidates saved to %s", len(candidates), out_path)
    return candidates


def run_detection(tic_ids: list[int], config: dict,
                  sde_threshold: float | None = None,
                  run_harmonic: bool = True) -> pd.DataFrame:
    df = run_bls_screen(
        tic_ids, config,
        sde_threshold=sde_threshold,
        run_harmonic=run_harmonic
    )

    if len(df) == 0:
        print("  ❌ No candidates detected. Pipeline stopped.")
        return df

    return run_tls_refinement(df, config)
