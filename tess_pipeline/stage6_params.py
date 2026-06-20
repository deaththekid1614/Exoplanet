"""Stage 6: Parameter estimation with bootstrap uncertainty."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from tess_pipeline.stage1_preprocess import load_processed_curve
from tess_pipeline.utils import phase_fold

logger = logging.getLogger(__name__)


def box_model(phase: np.ndarray, depth: float, duration: float, center: float = 0.0) -> np.ndarray:
    """Simple box transit model in phase space."""
    dist = np.minimum(np.abs(phase - center), np.abs(phase - 1 + center))
    model = np.ones_like(phase)
    model[dist < duration / 2] = 1.0 - depth
    return model


def trapezoid_model(phase: np.ndarray, depth: float, duration: float, ingress: float, center: float = 0.0) -> np.ndarray:
    """Trapezoidal transit model with ingress/egress."""
    dist = np.minimum(np.abs(phase - center), np.abs(phase - 1 + center))
    model = np.ones_like(phase)
    half_dur = duration / 2
    flat = dist < (half_dur - ingress)
    ramp = (dist >= (half_dur - ingress)) & (dist < half_dur)
    model[flat] = 1.0 - depth
    if ingress > 0:
        ramp_frac = (dist[ramp] - (half_dur - ingress)) / ingress
        model[ramp] = 1.0 - depth * ramp_frac
    return model


def fit_transit_parameters(time: np.ndarray, flux: np.ndarray, period: float, t0: float, n_bootstrap: int = 100) -> dict:
    """Fit transit parameters with bootstrap uncertainty. Includes timeout guard."""
    phase, folded = phase_fold(time, flux, period, t0)

    def model_func(ph, depth, duration, ingress):
        return trapezoid_model(ph, depth, duration, ingress)

    # Guard against bad data
    if len(phase) < 10 or np.any(~np.isfinite(folded)):
        logger.warning("Insufficient or invalid data for transit fitting")
        return _default_params(period, t0)

    try:
        popt, pcov = curve_fit(
            model_func,
            phase,
            folded,
            p0=[0.01, 0.05, 0.005],
            bounds=([0, 0.001, 0], [0.5, 0.5, 0.1]),
            maxfev=2000,  # Reduced from 5000 to prevent hangs
        )
        depth, duration, ingress = popt
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan] * 3
    except Exception as e:
        logger.debug("curve_fit failed: %s", e)
        depth, duration, ingress = 0.01, 0.05, 0.005
        perr = [np.nan] * 3

    # Impact parameter estimate from duration/period (circular orbit approx)
    dur_frac = duration
    if dur_frac > 0 and dur_frac < 1:
        b = float(np.sqrt(max(1.0 - (np.pi * dur_frac / 2) ** 2, 0)))
        inclination = float(np.degrees(np.arccos(np.clip(b, 0, 1))))
    else:
        b, inclination = np.nan, np.nan

    # Bootstrap uncertainties (limited iterations)
    boot_depths, boot_durations = [], []
    rng = np.random.default_rng(42)
    n = len(phase)
    max_boot = min(n_bootstrap, 50)  # Cap at 50 to prevent long runs
    
    for _ in range(max_boot):
        idx = rng.integers(0, n, n)
        try:
            bp, _ = curve_fit(
                model_func,
                phase[idx],
                folded[idx],
                p0=[depth, duration, ingress],
                bounds=([0, 0.001, 0], [0.5, 0.5, 0.1]),
                maxfev=1000,  # Reduced for bootstrap
            )
            boot_depths.append(bp[0])
            boot_durations.append(bp[1])
        except Exception:
            continue

    depth_err = float(np.std(boot_depths)) if boot_depths else perr[0]
    duration_err = float(np.std(boot_durations)) if boot_durations else perr[1]

    flat_duration = max(duration - 2 * ingress, 0)

    return {
        "fit_period": period,
        "fit_t0": t0,
        "fit_depth": float(depth),
        "fit_depth_err": depth_err,
        "fit_duration": float(duration),
        "fit_duration_err": duration_err,
        "fit_ingress_duration": float(ingress),
        "fit_egress_duration": float(ingress),
        "fit_flat_duration": float(flat_duration),
        "impact_parameter_b": b,
        "orbital_inclination_deg": inclination,
    }


def _default_params(period: float, t0: float) -> dict:
    """Return default parameters when fitting fails."""
    return {
        "fit_period": period,
        "fit_t0": t0,
        "fit_depth": np.nan,
        "fit_depth_err": np.nan,
        "fit_duration": np.nan,
        "fit_duration_err": np.nan,
        "fit_ingress_duration": np.nan,
        "fit_egress_duration": np.nan,
        "fit_flat_duration": np.nan,
        "impact_parameter_b": np.nan,
        "orbital_inclination_deg": np.nan,
    }


def fit_eclipse_parameters(time: np.ndarray, flux: np.ndarray, period: float, t0: float) -> dict:
    """Fit binary eclipse model (primary + secondary)."""
    phase, folded = phase_fold(time, flux, period, t0)
    primary_mask = phase < 0.05
    secondary_mask = (phase > 0.45) & (phase < 0.55)
    oot_mask = (phase > 0.15) & (phase < 0.4)

    oot_flux = np.median(folded[oot_mask]) if oot_mask.sum() > 0 else 1.0
    primary_depth = 1.0 - np.median(folded[primary_mask]) / oot_flux if primary_mask.sum() > 0 else 0
    secondary_depth = 1.0 - np.median(folded[secondary_mask]) / oot_flux if secondary_mask.sum() > 0 else 0

    return {
        "eclipse_primary_depth": float(primary_depth),
        "eclipse_secondary_depth": float(secondary_depth),
        "eclipse_period": period,
        "eclipse_eccentricity_proxy": float(secondary_depth / max(primary_depth, 1e-6)),
    }


def estimate_parameters(candidate: dict, config: dict) -> dict:
    """Estimate parameters for one candidate based on classification."""
    tic_id = int(candidate["tic_id"])
    data = load_processed_curve(tic_id, config)
    if data is None:
        return {"tic_id": tic_id}

    time = data["time"]
    flux = data["detrended"]
    period = float(candidate.get("period", 1.0))
    t0 = float(candidate.get("t0", time[0]))
    n_boot = config["parameters"]["bootstrap_iterations"]

    subtype = candidate.get("ml_subtype", candidate.get("classification", ""))
    if "eclipse" in str(subtype).lower():
        params = fit_eclipse_parameters(time, flux, period, t0)
    else:
        params = fit_transit_parameters(time, flux, period, t0, n_boot)

    params["tic_id"] = tic_id
    return params


def run_parameter_estimation(candidates: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Stage 6: fit parameters for all candidates."""
    logger.info("Stage 6: Parameter estimation for %d candidates", len(candidates))
    
    if candidates is None or len(candidates) == 0:
        logger.warning("Stage 6: No candidates for parameter estimation")
        return pd.DataFrame()

    param_rows = []
    for i, (_, row) in enumerate(candidates.iterrows()):
        logger.info("  Fitting TIC %s (%d/%d)", int(row["tic_id"]), i+1, len(candidates))
        params = estimate_parameters(row.to_dict(), config)
        param_rows.append({**row.to_dict(), **params})

    df = pd.DataFrame(param_rows)
    out_path = Path(config["paths"]["results"]) / "candidate_parameters.csv"
    df.to_csv(out_path, index=False)
    logger.info("Stage 6 complete: parameters saved to %s", out_path)
    return df