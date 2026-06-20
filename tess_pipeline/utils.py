"""Shared utilities for the TESS pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.constants import R_earth, R_jup, R_sun

logger = logging.getLogger(__name__)

# Physical constants for depth calculations
EARTH_RADIUS_SOLAR = (R_earth / R_sun).decompose().value
JUPITER_RADIUS_SOLAR = (R_jup / R_sun).decompose().value


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_tic_ids(path: str | Path) -> list[int]:
    """Load TIC IDs from CSV or plain text file.
    
    Handles:
    - Plain text: one TIC ID per line
    - CSV with header: 'tic_id' column
    - CSV without header: first column
    """
    path = Path(path)
    if not path.exists():
        logger.error("TIC ID file not found: %s", path)
        return []
    
    text = path.read_text().strip()
    if not text:
        logger.warning("TIC ID file is empty: %s", path)
        return []
    
    # Try CSV first if it looks like CSV (has commas)
    if "," in text:
        try:
            df = pd.read_csv(path)
            if "tic_id" in df.columns:
                return df["tic_id"].astype(int).tolist()
            elif len(df.columns) > 0:
                return df.iloc[:, 0].astype(int).tolist()
        except Exception as e:
            logger.warning("CSV parse failed for %s: %s. Falling back to plain text.", path, e)
    
    # Plain text: one TIC ID per line
    tic_ids = []
    for line in text.splitlines():
        line = line.strip()
        if line and line.isdigit():
            tic_ids.append(int(line))
    
    if not tic_ids:
        logger.warning("No valid TIC IDs found in %s", path)
    else:
        logger.info("Loaded %d TIC IDs from %s", len(tic_ids), path)
    
    return tic_ids


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def expected_transit_depth(r_planet_solar: float, r_star_solar: float) -> float:
    """Fractional transit depth: (R_planet / R_star)^2."""
    if r_star_solar <= 0:
        return np.nan
    return (r_planet_solar / r_star_solar) ** 2


def expected_earth_depth(r_star_solar: float) -> float:
    return expected_transit_depth(EARTH_RADIUS_SOLAR, r_star_solar)


def expected_jupiter_depth(r_star_solar: float) -> float:
    return expected_transit_depth(JUPITER_RADIUS_SOLAR, r_star_solar)


def blackbody_flux(wavelength_nm: float, teff_k: float) -> float:
    """Approximate blackbody flux (arbitrary units) at wavelength in nm."""
    wl = wavelength_nm * 1e-9 * u.m
    teff = teff_k * u.K
    # Wien's law peak scaling; simplified Planck proxy for relative band flux
    h = 6.626e-34
    c = 3e8
    k = 1.381e-23
    wl_m = wl.to(u.m).value
    t = teff.to(u.K).value
    x = h * c / (wl_m * k * t)
    if x > 700:
        return 0.0
    return (wl_m ** -5) / (np.expm1(x))


def phase_fold(time: np.ndarray, flux: np.ndarray, period: float, t0: float) -> tuple[np.ndarray, np.ndarray]:
    """Phase-fold light curve to [0, 1)."""
    phase = ((time - t0) / period) % 1.0
    order = np.argsort(phase)
    return phase[order], flux[order]


def score_from_ratio(observed: float, expected: float, tolerance: float = 0.5) -> float:
    """Score 0-1 based on how close observed is to expected."""
    if np.isnan(observed) or np.isnan(expected) or expected <= 0:
        return 0.0
    ratio = observed / expected
    log_dev = abs(np.log10(max(ratio, 1e-6)))
    return float(np.clip(1.0 - log_dev / tolerance, 0.0, 1.0))


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    """Append a single row to CSV, creating file with header if needed."""
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)