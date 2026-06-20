"""Stage 1: Preprocessing and quality control."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _savgol_window_length(time_days: np.ndarray, window_days: float) -> int:
    """Convert window in days to odd integer sample count."""
    dt = np.median(np.diff(time_days))
    if dt <= 0:
        return 11
    n = int(round(window_days / dt))
    n = max(n, 5)
    if n % 2 == 0:
        n += 1
    return min(n, len(time_days) - 1 if len(time_days) % 2 == 0 else len(time_days))


def detrend_savgol(time: np.ndarray, flux: np.ndarray, window_days: float, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay detrend; returns flux / trend (baseline ~1)."""
    if len(flux) < 7:
        return flux / np.nanmedian(flux)

    win = _savgol_window_length(time, window_days)
    if win >= len(flux):
        win = len(flux) - 1 if len(flux) % 2 == 0 else len(flux)
        if win < 5:
            return flux / np.nanmedian(flux)

    poly = min(polyorder, win - 1)
    trend = savgol_filter(flux, window_length=win, polyorder=poly)
    trend = np.where(trend > 0, trend, np.nanmedian(flux))
    return flux / trend


def clean_lightcurve(time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Remove NaNs and non-positive flux."""
    mask = np.isfinite(time) & np.isfinite(flux) & (flux > 0)
    if flux_err is not None:
        mask &= np.isfinite(flux_err)
    return time[mask], flux[mask], flux_err[mask] if flux_err is not None else None


def normalize_flux(flux: np.ndarray) -> np.ndarray:
    med = np.nanmedian(flux)
    return flux / med if med > 0 else flux


def inject_synthetic_transit(
    time: np.ndarray,
    flux: np.ndarray,
    period: float = 5.0,
    depth: float = 0.01,
    duration: float = 0.05,
    t0: float | None = None,
) -> np.ndarray:
    """Inject a box transit for validation."""
    flux = flux.copy()
    if t0 is None:
        t0 = time[len(time) // 4]
    for epoch in np.arange(t0, time[-1], period):
        in_transit = np.abs(((time - epoch + period / 2) % period) - period / 2) < duration / 2
        flux[in_transit] *= 1.0 - depth
    return flux


def validate_detrending(
    time: np.ndarray,
    flux: np.ndarray,
    config: dict,
) -> float:
    """Inject synthetic transits and measure recovery rate after detrending."""
    n_inject = config["preprocess"]["synthetic_validation_count"]
    recovered = 0
    window_days = config["preprocess"]["savgol_window_days"]

    for i in range(n_inject):
        depth = 0.005 + 0.01 * (i / max(n_inject - 1, 1))
        period = 2.0 + i * 1.5
        injected = inject_synthetic_transit(time, flux, period=period, depth=depth)
        detrended = detrend_savgol(time, normalize_flux(injected), window_days)
        folded = ((time - time[0]) / period) % 1.0
        bin_center = 0.0
        in_transit = folded < 0.05
        oot = folded > 0.15
        if in_transit.sum() > 0 and oot.sum() > 0:
            measured_depth = 1.0 - np.nanmedian(detrended[in_transit])
            if measured_depth > 0.5 * depth:
                recovered += 1

    return recovered / n_inject


def process_single_lightcurve(
    tic_id: int,
    lc_data: dict,
    config: dict,
) -> dict | None:
    """Process one light curve: clean, normalize, detrend, save."""
    time = np.asarray(lc_data["time"], dtype=float)
    flux = np.asarray(lc_data["flux"], dtype=float)
    flux_err = np.asarray(lc_data.get("flux_err", []), dtype=float) if lc_data.get("flux_err") is not None else None

    time, flux, flux_err = clean_lightcurve(time, flux, flux_err)
    if len(time) < 50:
        return None

    flux = normalize_flux(flux)
    window_days = config["preprocess"]["savgol_window_days"]
    polyorder = config["preprocess"]["savgol_polyorder"]
    detrended = detrend_savgol(time, flux, window_days, polyorder)

    out_dir = Path(config["paths"]["processed"])
    out_path = out_dir / f"{tic_id}.npz"
    save_kw = {"time": time, "flux": flux, "detrended": detrended}
    if flux_err is not None:
        save_kw["flux_err"] = flux_err
    np.savez_compressed(out_path, **save_kw)

    return {"tic_id": tic_id, "n_points": len(time), "path": str(out_path)}


def download_lightcurve(tic_id: int, sector: int | None, quality_bitmask: str):
    """Download light curve from MAST via lightkurve."""
    import lightkurve as lk

    if sector is not None:
        search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", sector=sector)
    else:
        search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")

    if len(search) == 0:
        return None

    lc = search[0].download(quality_bitmask=quality_bitmask)
    if lc is None:
        return None

    return {
        "time": lc.time.value,
        "flux": lc.flux.value,
        "flux_err": lc.flux_err.value if hasattr(lc, "flux_err") and lc.flux_err is not None else None,
    }


def _fetch_and_process(tic_id: int, sector: int | None, config: dict, skip_download: bool) -> dict | None:
    processed_path = Path(config["paths"]["processed"]) / f"{tic_id}.npz"
    if processed_path.exists():
        return {"tic_id": tic_id, "n_points": -1, "path": str(processed_path), "cached": True}

    if skip_download:
        return None

    try:
        lc_data = download_lightcurve(
            tic_id,
            sector,
            config["preprocess"]["quality_bitmask"],
        )
        if lc_data is None:
            return None
        return process_single_lightcurve(tic_id, lc_data, config)
    except Exception as e:
        logger.warning("Failed TIC %s: %s", tic_id, e)
        return None


def run_preprocess(
    tic_ids: list[int],
    config: dict,
    sector: int | None = None,
    skip_download: bool = False,
) -> pd.DataFrame:
    """Run Stage 1 on a list of TIC IDs in batches."""
    batch_size = config["preprocess"]["batch_size"]
    n_jobs = config["preprocess"]["n_jobs"]
    results = []

    logger.info("Stage 1: Processing %d light curves (batch_size=%d)", len(tic_ids), batch_size)

    for batch_start in tqdm(range(0, len(tic_ids), batch_size), desc="Preprocess batches"):
        batch = tic_ids[batch_start : batch_start + batch_size]
        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(_fetch_and_process)(tic_id, sector, config, skip_download) for tic_id in batch
        )
        results.extend([r for r in batch_results if r is not None])
        gc.collect()

    df = pd.DataFrame(results)
    manifest_path = Path(config["paths"]["processed"]) / "manifest.csv"
    df.to_csv(manifest_path, index=False)
    logger.info("Stage 1 complete: %d curves saved to %s", len(df), config["paths"]["processed"])
    return df


def load_processed_curve(tic_id: int, config: dict) -> dict | None:
    """Load a processed .npz file."""
    path = Path(config["paths"]["processed"]) / f"{tic_id}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}
