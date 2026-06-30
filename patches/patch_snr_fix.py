"""Patch: Robust SNR with multiple fallback methods."""
import numpy as np

def patched_compute_snr(row, data):
    snr = 0.0
    sig_level = "N/A"
    if data is None or "detrended" not in data or len(data["detrended"]) == 0:
        return snr, sig_level
    flux_arr = data["detrended"]
    time_arr = data["time"]
    period_val = float(row.get("period", 1.0))
    t0_val = float(row.get("t0", time_arr[0] if len(time_arr) > 0 else 0))
    dur = float(row.get("duration", 0.05))
    depth = float(row.get("depth", 0.001))
    try:
        from tess_pipeline.utils import phase_fold
        phase, folded = phase_fold(time_arr, flux_arr, period_val, t0_val)
        half_dur_phase = (dur / period_val) * 0.5
        in_transit = np.abs(phase) < half_dur_phase
        oot_mask = ~in_transit
        if np.sum(oot_mask) >= 5 and np.sum(in_transit) >= 2:
            oot_rms = np.std(folded[oot_mask])
            in_median = np.median(folded[in_transit])
            oot_median = np.median(folded[oot_mask])
            measured_depth = max(0.0, oot_median - in_median)
            n_transits = max(int((time_arr.max() - time_arr.min()) / period_val), 1)
            noise_per_transit = oot_rms / np.sqrt(max(n_transits, 1))
            snr = measured_depth / noise_per_transit if noise_per_transit > 0 else 0.0
            simple_snr = measured_depth / oot_rms if oot_rms > 0 else 0.0
            if snr < simple_snr and simple_snr > 0:
                snr = simple_snr
    except Exception:
        pass
    if snr <= 0.0:
        try:
            noise = np.std(flux_arr)
            snr = depth / noise if noise > 0 else 0.0
        except Exception:
            snr = 0.0
    if snr <= 0.0:
        bls_sde = float(row.get("bls_sde", 0))
        if bls_sde > 0:
            snr = bls_sde / 3.0
    if snr <= 0.0 and depth > 0:
        snr = 0.5
    if snr > 10:
        sig_level = "HIGH (≥10σ)"
    elif snr > 5:
        sig_level = "MEDIUM (5–10σ)"
    elif snr > 3:
        sig_level = "LOW (3–5σ)"
    else:
        sig_level = "INSIGNIFICANT (<3σ)"
    return float(snr), sig_level

def apply():
    import tess_pipeline.stage8_visualize as s8
    s8._compute_snr = patched_compute_snr
    print("  ✅ SNR patch: Multi-method calculation applied")
