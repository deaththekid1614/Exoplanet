"""
Stage 2.5: Harmonic Analysis — Fixed for real exoplanet detection.
Only sub-harmonics (P/2, P/3) can indicate wrong period.
Overtones (2P, 3P) are NORMAL for flat-bottomed planet transits.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

HARMONIC_FACTORS = [0.5, 1/3, 2.0, 3.0]


def _safe_scalar(val, default=0.0):
    """Convert array/list to scalar safely."""
    if val is None:
        return default
    if isinstance(val, (np.ndarray, list, tuple)):
        arr = np.asarray(val)
        if arr.size == 1:
            return float(arr.flat[0])
        return float(np.nanmax(arr))
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def analyze_harmonics(time, flux, period, epoch, duration, depth,
                      bls_power_original=None, bls_object=None,
                      duration_grid=None, n_phase_bins=100):
    """
    Fixed harmonic alias detection.
    
    CORRECT LOGIC:
    - 0.5x alias stronger than primary → true period is probably half (EB)
    - 0.333x alias stronger than primary → true period is probably third
    - 2x/3x overtones similar depth → NORMAL for flat planet transits, KEEP
    """
    results = {
        "harmonic_test_passed": True,
        "harmonic_flags": [],
        "harmonic_details": [],
        "primary_period_confidence": 1.0,
        "recommendation": "KEEP",
        "harmonic_flag": False,
        "harmonic_best_alias": None,
        "harmonic_best_sde": 0.0,
    }
    
    period_safe = _safe_scalar(period)
    epoch_safe = _safe_scalar(epoch)
    duration_safe = _safe_scalar(duration)
    depth_safe = _safe_scalar(depth)
    
    if depth_safe <= 0 or len(time) < 50:
        return results
    
    # Phase fold at primary period and compute primary depth
    phase_primary = ((time - epoch_safe) / period_safe) % 1.0
    phase_primary[phase_primary > 0.5] -= 1.0
    half_dur = (duration_safe / period_safe) / 2.0
    in_tr_primary = np.abs(phase_primary) < half_dur
    oot_primary = ~in_tr_primary
    
    if np.sum(in_tr_primary) < 3 or np.sum(oot_primary) < 10:
        return results
    
    primary_in = np.median(flux[in_tr_primary])
    primary_out = np.median(flux[oot_primary])
    primary_depth = max(0.0, primary_out - primary_in)
    
    if primary_depth < 1e-6:
        return results
    
    print(f"\n{'='*60}")
    print(f"  HARMONIC ANALYSIS")
    print(f"{'='*60}")
    print(f"  Primary Period:     {period_safe:.6f} d")
    print(f"  Primary Depth:      {primary_depth*1e6:.1f} ppm")
    print(f"  Duration:           {duration_safe*24:.2f} hr")
    print(f"{'-'*60}")
    
    best_alias_score = 0.0
    best_alias = None
    sub_harmonic_found = False
    
    for factor in HARMONIC_FACTORS:
        alias_period = period_safe * factor
        alias_label = f"{factor:.3f}x" if factor < 1 else f"{factor:.0f}x"
        is_sub_harmonic = factor < 1.0  # 0.5x or 0.333x
        
        # Phase fold at alias period
        phase_alias = ((time - epoch_safe) / alias_period) % 1.0
        phase_alias[phase_alias > 0.5] -= 1.0
        
        # Check number of expected transits
        n_transits_expected = int(np.ceil((time.max() - time.min()) / alias_period))
        if n_transits_expected < 2 and factor > 1:
            continue
        
        # Compute alias depth
        alias_dur_phase = (duration_safe / alias_period) / 2.0
        in_tr_alias = np.abs(phase_alias) < alias_dur_phase
        oot_alias = ~in_tr_alias
        
        if np.sum(in_tr_alias) < 3 or np.sum(oot_alias) < 10:
            continue
        
        alias_in = np.median(flux[in_tr_alias])
        alias_out = np.median(flux[oot_alias])
        alias_depth = max(0.0, alias_out - alias_in)
        
        # Signal-to-noise estimate
        alias_noise = np.std(flux[oot_alias])
        alias_snr = alias_depth / (alias_noise + 1e-10) if alias_noise > 0 else 0
        
        # Depth ratio
        depth_ratio = alias_depth / (primary_depth + 1e-10)
        
        # === CORRECT LOGIC ===
        status = "OK"
        
        if is_sub_harmonic:
            # SUB-HARMONIC: if stronger than primary, true period is probably the alias
            if depth_ratio > 0.8 and alias_snr > 3.0:
                status = "ALIAS_STRONG"
                results["harmonic_flags"].append(f"{alias_label}_alias_strong")
                results["harmonic_test_passed"] = False
                results["recommendation"] = "REVIEW"
                results["harmonic_flag"] = True
                sub_harmonic_found = True
            elif depth_ratio > 0.5 and alias_snr > 2.0:
                status = "ALIAS_WEAK"
                results["harmonic_flags"].append(f"{alias_label}_alias_weak")
                results["harmonic_flag"] = True
        else:
            # OVERTONE (2x, 3x): Similar depth is NORMAL for planets
            # Only flag if it's suspiciously strong (rare)
            if depth_ratio > 1.2 and alias_snr > 5.0:
                status = "OVERTONE_STRONG"
                # Don't auto-reject, just note it
                results["harmonic_flags"].append(f"{alias_label}_overtone_strong")
            elif depth_ratio > 0.85:
                status = "OVERTONE_OK"
                # Normal for flat transits — no action needed
        
        detail = {
            "factor": float(factor),
            "alias_period": float(alias_period),
            "alias_depth": float(alias_depth),
            "depth_ratio": float(depth_ratio),
            "alias_snr": float(alias_snr),
            "status": status,
            "n_transits_expected": int(n_transits_expected)
        }
        results["harmonic_details"].append(detail)
        
        # Track best alias for reporting (only sub-harmonics matter)
        alias_score = depth_ratio * min(alias_snr / 5.0, 1.0) if is_sub_harmonic else 0
        if alias_score > best_alias_score:
            best_alias_score = alias_score
            best_alias = detail
        
        print(f"  {alias_label:>8} | P={alias_period:.6f}d | "
              f"Depth={alias_depth*1e6:.1f}ppm ({depth_ratio:.2f}x) | "
              f"SNR={alias_snr:.2f} | [{status}]")
    
    # Final confidence and recommendation
    if sub_harmonic_found and best_alias_score > 0.5:
        results["primary_period_confidence"] = max(0.1, 1.0 - best_alias_score)
        results["recommendation"] = "REJECT" if best_alias_score > 1.5 else "REVIEW"
        results["harmonic_best_alias"] = float(best_alias["alias_period"])
        results["harmonic_best_sde"] = float(best_alias["alias_snr"])
        print(f"\n  ⚠️  SUB-HARMONIC ALIAS DETECTED!")
        print(f"      True period may be: {best_alias['factor']:.3f}x of detected")
        print(f"      Alias score: {best_alias_score:.3f}")
        print(f"      Confidence in primary: {results['primary_period_confidence']:.2%}")
    else:
        print(f"\n  ✅ Primary period is likely genuine.")
        print(f"      Confidence: {results['primary_period_confidence']:.2%}")
        if results["harmonic_flags"]:
            print(f"      Notes: {', '.join(results['harmonic_flags'])} (non-critical)")
    
    print(f"{'='*60}")
    print(f"  Result: {results['recommendation']}")
    print(f"  Flags:  {results['harmonic_flags'] if results['harmonic_flags'] else 'None'}")
    print(f"{'='*60}\n")
    
    return results


# Terminal standalone test
if __name__ == "__main__":
    print("=" * 70)
    print("  HARMONIC ANALYSIS MODULE — STANDALONE TEST")
    print("=" * 70)
    
    np.random.seed(42)
    t = np.linspace(0, 30, 3000)
    true_period = 3.5
    true_epoch = 0.5
    true_depth = 0.001
    true_dur = 0.05
    
    flux = np.ones_like(t)
    phase = ((t - true_epoch) / true_period) % 1.0
    phase[phase > 0.5] -= 1.0
    flux[np.abs(phase) < true_dur/2] -= true_depth
    flux += np.random.normal(0, 0.0003, len(t))
    
    results = analyze_harmonics(t, flux, true_period, true_epoch, 
                                true_dur * true_period, true_depth)
    
    print(f"\nFinal: {results['recommendation']}")
    print(f"harmonic_flag: {results['harmonic_flag']}")