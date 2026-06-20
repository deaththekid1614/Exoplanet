"""Generate synthetic light curves for demo/testing without MAST access."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# STELLAR VARIABILITY
# ═══════════════════════════════════════════════════════════════════════════════

def generate_stellar_variability(time: np.ndarray, amp: float = 0.003, period: float = 5.0,
                                  seed: int | None = None) -> np.ndarray:
    """Simulate realistic stellar variability (rotation + granulation)."""
    rng = np.random.default_rng(seed)
    flux = np.ones_like(time)
    
    # Rotation (sinusoidal)
    flux += amp * np.sin(2 * np.pi * time / period + rng.uniform(0, 2*np.pi))
    
    # Granulation (red noise)
    gran_amp = amp * 0.3
    gran_time = np.linspace(0, time[-1], len(time))
    gran_noise = np.cumsum(rng.normal(0, gran_amp/10, len(time)))
    gran_noise -= np.mean(gran_noise)
    flux += gran_noise * 0.5
    
    return flux


# ═══════════════════════════════════════════════════════════════════════════════
# MANDEL-AGOL TRANSIT MODEL (realistic limb-darkened shape)
# ═══════════════════════════════════════════════════════════════════════════════

def _mandel_agol_z(rp: float, z: np.ndarray) -> np.ndarray:
    """Mandel-Agol transit model for uniform source (simplified)."""
    # z = projected separation / stellar radius
    # rp = planet radius / stellar radius
    mu = np.zeros_like(z)
    
    # Outside transit
    out = z > (1.0 + rp)
    mu[out] = 1.0
    
    # Full transit
    full = z <= (1.0 - rp)
    mu[full] = 1.0 - rp**2
    
    # Ingress/egress (quadratic approximation)
    ingress = (z > (1.0 - rp)) & (z <= (1.0 + rp))
    k0 = np.arccos((1.0 - rp**2 + z[ingress]**2) / (2.0 * z[ingress]))
    k1 = np.arccos((rp**2 + z[ingress]**2 - 1.0) / (2.0 * rp * z[ingress]))
    mu[ingress] = 1.0 - (rp**2 * k1 + k0 - 0.5 * np.sqrt(
        np.maximum(0.0, 4.0 * z[ingress]**2 - (1.0 + z[ingress]**2 - rp**2)**2)
    )) / np.pi
    
    return mu


def inject_transit(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    depth: float,
    duration: float,
    t0: float,
    noise: float = 0.0003,
    limb_darkening: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """Inject realistic Mandel-Agol transit with noise."""
    rng = np.random.default_rng(seed)
    flux = flux.copy()
    rp = np.sqrt(depth)  # Planet radius in stellar radii
    
    # Duration in phase units
    dur_phase = duration / period
    
    epoch_count = 0
    for epoch in np.arange(t0, time[-1] + period, period):
        phase = ((time - epoch) / period + 0.5) % 1.0 - 0.5  # Centered on 0
        z = np.abs(phase) / (dur_phase / 2.0)  # Normalized separation
        
        if limb_darkening:
            mu = _mandel_agol_z(rp, z)
        else:
            # Box transit fallback
            mu = np.where(z <= 1.0, 1.0 - depth, 1.0)
        
        flux *= mu
        epoch_count += 1
    
    # Realistic TESS noise (white + correlated)
    white_noise = rng.normal(0, noise, len(flux))
    # Add slight systematic drift
    systematic = np.sin(2 * np.pi * time / 0.5) * noise * 0.5
    flux += white_noise + systematic
    
    return flux


# ═══════════════════════════════════════════════════════════════════════════════
# ECLIPSE BINARY (with secondary eclipse)
# ═══════════════════════════════════════════════════════════════════════════════

def inject_eclipse(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    primary_depth: float,
    secondary_depth: float,
    duration: float,
    t0: float,
    noise: float = 0.0003,
    seed: int | None = None,
) -> np.ndarray:
    """Inject binary eclipse with primary at phase 0 and secondary at phase 0.5."""
    rng = np.random.default_rng(seed)
    flux = flux.copy()
    half_dur = duration / 2.0
    
    for epoch in np.arange(t0, time[-1] + period, period):
        phase = ((time - epoch) / period) % 1.0
        phase[phase > 0.5] -= 1.0  # Center on 0
        
        # Primary eclipse (deeper, at phase 0)
        in_primary = np.abs(phase) < half_dur
        flux[in_primary] *= 1.0 - primary_depth
        
        # Secondary eclipse (shallower, at phase 0.5)
        phase_sec = ((time - epoch) / period + 0.5) % 1.0
        phase_sec[phase_sec > 0.5] -= 1.0
        in_secondary = np.abs(phase_sec) < half_dur * 0.8
        flux[in_secondary] *= 1.0 - secondary_depth
    
    # Noise
    flux += rng.normal(0, noise, len(flux))
    return flux


# ═══════════════════════════════════════════════════════════════════════════════
# BLEND (deep transit from background eclipsing binary)
# ═══════════════════════════════════════════════════════════════════════════════

def inject_blend(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    depth: float,
    duration: float,
    t0: float,
    noise: float = 0.0003,
    seed: int | None = None,
) -> np.ndarray:
    """Inject blend — deep V-shaped event (eclipsing binary signature)."""
    rng = np.random.default_rng(seed)
    flux = flux.copy()
    
    for epoch in np.arange(t0, time[-1] + period, period):
        phase = ((time - epoch) / period + 0.5) % 1.0 - 0.5
        # V-shaped ingress/egress (shallower slopes than planet)
        z = np.abs(phase) / (duration / period / 2.0)
        # Linear drop for blend
        mu = np.where(z <= 1.0, 1.0 - depth * (1.0 - z), 1.0)
        mu = np.clip(mu, 1.0 - depth, 1.0)
        flux *= mu
    
    flux += rng.normal(0, noise, len(flux))
    return flux


# ═══════════════════════════════════════════════════════════════════════════════
# STELLAR ACTIVITY (spots + flares)
# ═══════════════════════════════════════════════════════════════════════════════

def inject_stellar_activity(
    time: np.ndarray,
    flux: np.ndarray,
    spot_period: float,
    spot_amp: float,
    flare_prob: float = 0.001,
    seed: int | None = None,
) -> np.ndarray:
    """Inject realistic stellar activity (rotating spots + occasional flares)."""
    rng = np.random.default_rng(seed)
    flux = flux.copy()
    
    # Rotating spots
    flux += spot_amp * np.sin(2 * np.pi * time / spot_period + rng.uniform(0, 2*np.pi))
    
    # Occasional flares (sharp positive spikes)
    n_flares = int(len(time) * flare_prob)
    flare_idx = rng.choice(len(time), n_flares, replace=False)
    for idx in flare_idx:
        width = rng.integers(3, 10)
        start = max(0, idx - width//2)
        end = min(len(time), idx + width//2)
        flare_amp = rng.uniform(0.005, 0.02)
        flare_shape = np.exp(-np.linspace(-2, 2, end-start)**2)
        flux[start:end] += flare_amp * flare_shape
    
    return flux


# ═══════════════════════════════════════════════════════════════════════════════
# HARMONIC ALIAS TARGET (for testing harmonic analysis)
# ═══════════════════════════════════════════════════════════════════════════════

def inject_harmonic_alias(
    time: np.ndarray,
    flux: np.ndarray,
    true_period: float,
    alias_period: float,
    depth: float,
    duration: float,
    t0: float,
    noise: float = 0.0003,
    seed: int | None = None,
) -> np.ndarray:
    """Inject transit at true_period but also add signal at alias_period (P/2).
    This tests if harmonic analysis correctly flags the alias."""
    rng = np.random.default_rng(seed)
    
    # First inject the real transit
    flux = inject_transit(time, flux, true_period, depth, duration, t0, noise=noise*0.7, seed=seed)
    
    # Then add weaker signal at alias period (P/2) to confuse BLS
    flux = inject_transit(time, flux, alias_period, depth * 0.6, duration * 0.8, 
                          t0 + true_period*0.25, noise=noise*0.3, seed=seed)
    
    return flux


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DEMO DATASET GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_demo_dataset(config: dict, n_curves: int = 200, seed: int = 42) -> tuple[list[int], pd.DataFrame]:
    """Generate synthetic TESS-like light curves with diverse labels for testing all features."""
    rng = np.random.default_rng(seed)
    n_days = 27.0
    cadence = 2.0 / 1440  # 2-minute cadence in days
    time = np.arange(0, n_days, cadence)

    tic_ids = list(range(1000001, 1000001 + n_curves))
    labels = []
    
    out_dir = Path(config["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, tic_id in enumerate(tic_ids):
        # Base stellar variability (low amplitude for detectability)
        flux = generate_stellar_variability(
            time, 
            amp=rng.uniform(0.0003, 0.0015), 
            period=rng.uniform(8, 25),
            seed=seed + i
        )

        roll = rng.random()
        
        if roll < 0.18:
            # ═══════════════════════════════════════════════════════════════
            # PLANET TRANSIT — realistic, detectable
            # ═══════════════════════════════════════════════════════════════
            period = rng.uniform(2.5, 12.0)
            depth = rng.uniform(0.008, 0.035)  # 0.8% to 3.5% depth
            duration = rng.uniform(0.03, 0.08)  # 0.7 to 2 hours
            t0 = rng.uniform(1, 10)
            flux = inject_transit(time, flux, period, depth, duration, t0, 
                                  noise=0.00025, limb_darkening=True, seed=seed+i)
            label = "planet"
            
        elif roll < 0.28:
            # ═══════════════════════════════════════════════════════════════
            # ECLIPSING BINARY — with secondary eclipse
            # ═══════════════════════════════════════════════════════════════
            period = rng.uniform(2.0, 8.0)
            primary_depth = rng.uniform(0.04, 0.12)
            secondary_depth = rng.uniform(0.01, 0.04)
            duration = rng.uniform(0.03, 0.07)
            t0 = rng.uniform(1, 8)
            flux = inject_eclipse(time, flux, period, primary_depth, secondary_depth, 
                                  duration, t0, noise=0.0003, seed=seed+i)
            label = "eclipse"
            
        elif roll < 0.38:
            # ═══════════════════════════════════════════════════════════════
            # BLEND — deep V-shaped event
            # ═══════════════════════════════════════════════════════════════
            period = rng.uniform(2.5, 7.0)
            depth = rng.uniform(0.06, 0.18)
            duration = rng.uniform(0.04, 0.09)
            t0 = rng.uniform(1, 8)
            flux = inject_blend(time, flux, period, depth, duration, t0, 
                                noise=0.0003, seed=seed+i)
            label = "blend"
            
        elif roll < 0.48:
            # ═══════════════════════════════════════════════════════════════
            # HARMONIC ALIAS — tests harmonic analysis module
            # ═══════════════════════════════════════════════════════════════
            true_period = rng.uniform(6.0, 14.0)
            alias_period = true_period / 2.0
            depth = rng.uniform(0.01, 0.03)
            duration = rng.uniform(0.04, 0.07)
            t0 = rng.uniform(2, 8)
            flux = inject_harmonic_alias(time, flux, true_period, alias_period, 
                                         depth, duration, t0, noise=0.0003, seed=seed+i)
            label = "harmonic_alias"
            
        elif roll < 0.65:
            # ═══════════════════════════════════════════════════════════════
            # STELLAR ACTIVITY — spots + flares
            # ═══════════════════════════════════════════════════════════════
            spot_period = rng.uniform(4, 15)
            spot_amp = rng.uniform(0.002, 0.008)
            flux = inject_stellar_activity(time, flux, spot_period, spot_amp, 
                                           flare_prob=0.0005, seed=seed+i)
            label = "stellar_activity"
            
        else:
            # ═══════════════════════════════════════════════════════════════
            # NOISE / FALSE POSITIVE — flat with noise
            # ═══════════════════════════════════════════════════════════════
            label = "false_positive" if roll < 0.82 else "noise"

        # Normalize to median = 1.0
        flux = flux / np.nanmedian(flux)
        
        # Save to NPZ (detrended = flux for demo)
        out_path = out_dir / f"{tic_id}.npz"
        np.savez_compressed(
            out_path,
            time=time,
            flux=flux,
            detrended=flux,
            flux_err=np.full_like(flux, 0.0003)
        )
        
        labels.append({
            "tic_id": tic_id,
            "label": label,
            "true_period": period if label in ["planet", "eclipse", "blend", "harmonic_alias"] else None,
            "true_depth": depth if label in ["planet", "blend", "harmonic_alias"] else 
                          (primary_depth if label == "eclipse" else None),
        })

    labels_df = pd.DataFrame(labels)
    train_dir = Path(config["paths"]["training"])
    train_dir.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(train_dir / "training_labels.csv", index=False)

    logger.info(
        "Generated %d demo curves: "
        "18%% planets | 10%% eclipses | 10%% blends | 10%% harmonic_aliases | "
        "17%% stellar_activity | 35%% noise/FP",
        n_curves
    )
    return tic_ids, labels_df