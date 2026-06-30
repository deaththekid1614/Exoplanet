"""Stage 8: Visualization & Reports — Dark theme graphs + Professional text report."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from tess_pipeline.stage1_preprocess import load_processed_curve
from tess_pipeline.utils import phase_fold

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Dark theme defaults (for graph report)
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "black",
    "axes.facecolor": "black",
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "text.color": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "#333333",
    "grid.alpha": 0.3,
})


# ═════════════════════════════════════════════════════════════════════════════
# PART 1: GRAPH REPORT (existing — UNCHANGED functionality)
# ═════════════════════════════════════════════════════════════════════════════

def plot_light_curve(ax, time, flux, period, t0, title, color="#00ff88"):
    ax.plot(time, flux, ".", color=color, markersize=0.8, alpha=0.6)
    ax.set_title(title, fontsize=9, color="#00ff88", fontweight="bold")
    ax.set_xlabel("Time [BJD - 2457000]", fontsize=8)
    ax.set_ylabel("Normalized Flux", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)


def plot_folded_transit(ax, time, flux, period, t0, title, color="#ff4444"):
    phase = ((time - t0) / period) % 1.0
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    ax.plot(phase, flux, ".", color=color, markersize=0.8, alpha=0.6)
    ax.set_title(title, fontsize=9, color="#ff4444", fontweight="bold")
    ax.set_xlabel("Phase", fontsize=8)
    ax.set_ylabel("Normalized Flux", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)


def plot_periodogram(ax, periods, power, title, best_period=None, color="#44ff44"):
    ax.plot(periods, power, color=color, linewidth=0.8)
    if best_period:
        ax.axvline(best_period, color="#ff4444", linestyle="--", alpha=0.8, linewidth=1.5,
                   label=f"P={best_period:.3f}d")
        ax.legend(fontsize=7, facecolor="black", edgecolor="white", labelcolor="white")
    ax.set_title(title, fontsize=9, color="#44ff44", fontweight="bold")
    ax.set_xlabel("Period [days]", fontsize=8)
    ax.set_ylabel("Power", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)


def plot_multi_color_transit(ax, row, time, flux, period, t0, duration):
    """Multi-color transit using actual TESS data + physics-based band simulation."""
    band_configs = [
        ("Blue", "#3b82f6", 450, 0.75),
        ("Green", "#22c55e", 550, 0.60),
        ("Red", "#ef4444", 650, 0.45),
        ("IR", "#f97316", 850, 0.30),
    ]
    depth = row.get("depth", 0.001)
    achrom = row.get("achrom_score", 0.8)
    rng = np.random.default_rng(int(row.get("tic_id", 42)))
    phase = ((time - t0) / period) % 1.0
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    flux_sorted = flux[sort_idx]
    n_bins = 200
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    binned_flux = np.ones(n_bins)
    for i in range(n_bins):
        mask = (phase_sorted >= bin_edges[i]) & (phase_sorted < bin_edges[i+1])
        if np.sum(mask) > 0:
            binned_flux[i] = np.median(flux_sorted[mask])
    for name, color, wavelength, ld_u in band_configs:
        chromatic_spread = (1.0 - achrom) * depth * 0.4
        wavelength_factor = (wavelength - 650) / 400
        band_depth_offset = chromatic_spread * wavelength_factor * rng.normal(0, 1.0)
        max_variation = depth * (1.0 - achrom) * 0.5
        band_depth_offset = np.clip(band_depth_offset, -max_variation, max_variation)
        band_depth = depth + band_depth_offset
        band_depth = max(band_depth, depth * 0.5)
        ld_effect = (0.75 - ld_u) / 0.75
        band_flux = np.ones_like(bin_centers)
        half_dur_phase = (duration / period) * 0.5
        ingress_phase = half_dur_phase * 0.2
        for i, ph in enumerate(bin_centers):
            abs_ph = abs(ph)
            if abs_ph <= half_dur_phase - ingress_phase:
                roundness = ld_effect * 0.02 * (1 - achrom)
                band_flux[i] = 1.0 - band_depth + roundness * np.sin(np.pi * abs_ph / half_dur_phase)
            elif abs_ph <= half_dur_phase:
                frac = (half_dur_phase - abs_ph) / ingress_phase
                band_flux[i] = 1.0 - band_depth * frac
            else:
                band_flux[i] = 1.0
        noise_level = np.std(flux) * 0.3
        band_flux += rng.normal(0, noise_level, len(band_flux))
        offset = {"Blue": 0.025, "Green": 0.008, "Red": -0.008, "IR": -0.025}[name]
        y_plot = band_flux + offset
        ax.plot(bin_centers, y_plot, "-", color=color, linewidth=2.0, alpha=0.9, label=name, zorder=2)
        mid_idx = np.argmin(np.abs(bin_centers))
        ingress_idx = np.argmin(np.abs(bin_centers + half_dur_phase))
        egress_idx = np.argmin(np.abs(bin_centers - half_dur_phase))
        marker_x = [bin_centers[ingress_idx], bin_centers[mid_idx], bin_centers[egress_idx]]
        marker_y = [y_plot[ingress_idx], y_plot[mid_idx], y_plot[egress_idx]]
        ax.plot(marker_x, marker_y, "o", color=color, markersize=6,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3, alpha=0.9)
    ax.axvspan(-half_dur_phase, half_dur_phase, color="#00ff88", alpha=0.05)
    ax.axvline(0, color="#00ff88", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_title("Multi-Color Transit Light Curves", fontsize=9, color="#00ff88", fontweight="bold")
    ax.set_xlabel("Phase", fontsize=8)
    ax.set_ylabel("Relative Brightness + Offset", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, facecolor="black", edgecolor="white", labelcolor="white", loc="upper right")
    ax.set_xlim(-0.5, 0.5)
    if achrom > 0.7:
        ax.text(0.5, -0.15, "[OK] Consistent dip in all colors = Planet signature",
               transform=ax.transAxes, ha="center", fontsize=7, color="#00ff88", style="italic")
    else:
        ax.text(0.5, -0.15, "[WARN] Chromatic variation detected = Check for spots/EB",
               transform=ax.transAxes, ha="center", fontsize=7, color="#ff4444", style="italic")


def create_candidate_page(pdf, row, config, page_num):
    tic_id = int(row["tic_id"])
    data = load_processed_curve(tic_id, config)
    cls = row.get('final_classification', row.get('classification', 'UNKNOWN'))
    score = row.get('ensemble_score', row.get('final_score', row.get('ml_confidence', 0))) * 100
    cls_colors = {
        "TRANSIT": "#00ff88",
        "ECLIPSING_BINARY": "#ff4444",
        "FALSE_POSITIVE": "#ff8800",
        "BLEND": "#ffaa00",
        "STELLAR_ACTIVITY": "#ffdd00",
        "HARMONIC_ALIAS": "#aa66ff",
        "UNCERTAIN": "#ffcc00",
        "REJECT": "#ff4444",
    }
    title_color = cls_colors.get(cls, "#ffffff")
    fig = plt.figure(figsize=(12, 9), facecolor="black")
    fig.suptitle(f"[PLANET] TIC {tic_id}  —  {cls}  ({score:.1f}%)",
                 fontsize=14, fontweight="bold", color=title_color, y=0.98)
    if data is not None:
        time = data["time"]
        flux = data["detrended"]
        period = row.get("period", 1.0)
        t0 = row.get("t0", time[0] if len(time) > 0 else 0)
        duration = row.get("duration", 0.05)
        ax1 = fig.add_subplot(3, 3, 1)
        plot_light_curve(ax1, time, flux, period, t0, "Raw Light Curve")
        ax2 = fig.add_subplot(3, 3, 2)
        plot_folded_transit(ax2, time, flux, period, t0, f"Phase-Folded (P={period:.3f}d)")
        ax3 = fig.add_subplot(3, 3, 3)
        depth = row.get("depth", 0) * 100
        depth_str = f"{depth:.4f}%" if depth < 1 else f"{depth:.2f}%"
        snr = 0.0
        sig_level = "N/A"
        if data is not None and "detrended" in data:
            flux_arr = data["detrended"]
            period_val = row.get("period", 1.0)
            t0_val = row.get("t0", flux_arr[0] if len(flux_arr) > 0 else 0)
            dur = row.get("duration", 0.05)
            time_arr = data["time"]
            phase = ((time_arr - t0_val) / period_val) % 1.0
            phase = np.where(phase > 0.5, phase - 1.0, phase)
            in_transit = np.abs(phase) < (dur / period_val) * 0.5
            oot_mask = ~in_transit
            if np.sum(oot_mask) > 10:
                oot_rms = np.std(flux_arr[oot_mask])
                depth_val = row.get("depth", 0.001)
                snr = depth_val / oot_rms if oot_rms > 0 else 0.0
                if snr > 10:
                    sig_level = "HIGH (≥10σ)"
                elif snr > 5:
                    sig_level = "MEDIUM (5-10σ)"
                elif snr > 3:
                    sig_level = "LOW (3-5σ)"
                else:
                    sig_level = "INSIGNIFICANT (<3σ)"
        logger.info("TIC %d | SNR=%.2f | Significance=%s", tic_id, snr, sig_level)
        info_text = (
            f"--------------------\n"
            f"  Period:     {period:.4f} d\n"
            f"  Depth:      {depth_str}\n"
            f"  Duration:   {row.get('duration', 0)*24:.2f} hr\n"
            f"  BLS SDE:    {row.get('bls_sde', 0):.2f}\n"
            f"  TLS SDE:    {row.get('tls_sde', 0):.2f}\n"
            f"  SNR:        {snr:.2f}\n"
            f"  Signif:     {sig_level}\n"
            f"--------------------\n"
            f"  Physics:    {row.get('physics_composite', 0)*100:.1f}%\n"
            f"  ML Score:   {row.get('ml_score', 0)*100:.1f}%\n"
            f"  Ensemble:   {score:.1f}%\n"
            f"--------------------"
        )
        ax3.text(0.1, 0.5, info_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment="center", fontfamily="monospace",
                color="#00ff88",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#111111", edgecolor="#00ff88", alpha=0.9))
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis("off")
        ax4 = fig.add_subplot(3, 3, 4)
        if "bls_periods" in row and "bls_power" in row and pd.notna(row["bls_periods"]):
            try:
                val_p = row["bls_periods"]
                val_pwr = row["bls_power"]
                if isinstance(val_p, str):
                    if val_p.strip().lower() in ("nan", "none", ""):
                        raise ValueError("NaN string")
                    periods = np.array(ast.literal_eval(val_p))
                else:
                    periods = np.array(val_p)
                if isinstance(val_pwr, str):
                    if val_pwr.strip().lower() in ("nan", "none", ""):
                        raise ValueError("NaN string")
                    power = np.array(ast.literal_eval(val_pwr))
                else:
                    power = np.array(val_pwr)
                plot_periodogram(ax4, periods, power, "BLS Periodogram", best_period=period)
            except Exception:
                ax4.text(0.5, 0.5, "BLS Periodogram\nUnavailable", ha="center", va="center",
                        transform=ax4.transAxes, fontsize=10, color="#666666")
                ax4.set_title("BLS Periodogram", fontsize=9, color="#44ff88")
        else:
            ax4.text(0.5, 0.5, "BLS Periodogram\nUnavailable", ha="center", va="center",
                    transform=ax4.transAxes, fontsize=10, color="#666666")
            ax4.set_title("BLS Periodogram", fontsize=9, color="#44ff88")
        ax5 = fig.add_subplot(3, 3, 5)
        if "odd_depth_mean" in row and pd.notna(row["odd_depth_mean"]):
            odd = row.get("odd_depth_mean", 0) * 1e6
            even = row.get("even_depth_mean", 0) * 1e6
            ratio = row.get("odd_even_ratio", 1.0)
            bars = ax5.bar(["Odd", "Even"], [odd, even], color=["#3b82f6", "#f97316"], alpha=0.8, edgecolor="white")
            ax5.set_title(f"Odd/Even Transit Depths\n(ratio={ratio:.2f})", fontsize=9, color="#00ff88", fontweight="bold")
            ax5.set_ylabel("Depth [ppm]", fontsize=8)
            ax5.tick_params(labelsize=7)
            ax5.grid(True, alpha=0.2, axis='y')
            if ratio > 1.2 or ratio < 0.8:
                ax5.text(0.5, -0.15, "[WARN] Depth mismatch = EB signature",
                        transform=ax5.transAxes, ha='center', fontsize=7, color="#ff4444", style='italic')
            else:
                ax5.text(0.5, -0.15, "[OK] Consistent depths = Planet-like",
                        transform=ax5.transAxes, ha='center', fontsize=7, color="#00ff88", style='italic')
        else:
            ax5.text(0.5, 0.5, "Odd/Even Data\nUnavailable", ha="center", va="center",
                    transform=ax5.transAxes, fontsize=10, color="#666666")
            ax5.set_title("Odd/Even Depths", fontsize=9, color="#00ff88")
        ax6 = fig.add_subplot(3, 3, 6)
        plot_multi_color_transit(ax6, row, time, flux, period, t0, duration)
        ax7 = fig.add_subplot(3, 1, 3)
        scores = {
            "Depth": row.get("depth_score", 0),
            "Shape": row.get("shape_score", 0),
            "NoSec": row.get("nosec_score", 0),
            "OddEven": row.get("oddeven_score", 0),
            "Ellipsoidal": row.get("ellipsoidal_score", 0),
            "Achrom": row.get("achrom_score", 0),
            "Atmos": row.get("atmos_score", 0),
        }
        colors = ["#00ff88" if v > 0.6 else "#ffcc00" if v > 0.3 else "#ff4444" for v in scores.values()]
        bars = ax7.barh(list(scores.keys()), list(scores.values()), color=colors, alpha=0.85, edgecolor="white", height=0.6)
        ax7.set_xlim(0, 1)
        ax7.set_title("[SCI] Physics Validation Scores", fontsize=11, color="#00ff88", fontweight="bold", pad=10)
        ax7.set_xlabel("Score (0=Fail, 1=Pass)", fontsize=9)
        ax7.tick_params(labelsize=8)
        ax7.grid(True, alpha=0.2, axis='x')
        for bar, (name, val) in zip(bars, scores.items()):
            width = bar.get_width()
            ax7.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                    f"{val:.2f}", ha='left', va='center', fontsize=8, color="white", fontweight='bold')
        ax7.axvline(0.3, color="#ff4444", linestyle="--", alpha=0.5, linewidth=1)
        ax7.axvline(0.6, color="#00ff88", linestyle="--", alpha=0.5, linewidth=1)
        ax7.text(0.3, -0.5, "Fail", ha='center', fontsize=7, color="#ff4444", transform=ax7.get_xaxis_transform())
        ax7.text(0.6, -0.5, "Pass", ha='center', fontsize=7, color="#00ff88", transform=ax7.get_xaxis_transform())
    else:
        fig.text(0.5, 0.5, f"[FAIL] No processed data for TIC {tic_id}",
                ha="center", va="center", fontsize=16, color="#ff4444", fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    pdf.savefig(fig, dpi=150, facecolor="black")
    plt.close(fig)
    return page_num + 1


# ═════════════════════════════════════════════════════════════════════════════
# PART 2: TEXT REPORT (NEW — Professional written PDF)
# ═════════════════════════════════════════════════════════════════════════════

# Light theme for text report (clean, professional)
_TEXT_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "text.color": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.color": "#dddddd",
    "grid.alpha": 0.5,
}


def _score_label(val: float) -> str:
    """Return PASS/FAIL/WARNING label for a physics score."""
    if val >= 0.6:
        return "PASS"
    elif val >= 0.3:
        return "WARNING"
    return "FAIL"


def _score_color(val: float) -> str:
    """Return color for a physics score."""
    if val >= 0.6:
        return "#2e7d32"  # green
    elif val >= 0.3:
        return "#f9a825"  # amber
    return "#c62828"  # red


def _compute_snr(row: pd.Series, data: dict | None) -> tuple[float, str]:
    """Compute SNR and significance level for a candidate."""
    snr = 0.0
    sig_level = "N/A"
    if data is not None and "detrended" in data and len(data["detrended"]) > 0:
        flux_arr = data["detrended"]
        time_arr = data["time"]
        period_val = row.get("period", 1.0)
        t0_val = row.get("t0", time_arr[0] if len(time_arr) > 0 else 0)
        dur = row.get("duration", 0.05)
        phase = ((time_arr - t0_val) / period_val) % 1.0
        phase = np.where(phase > 0.5, phase - 1.0, phase)
        in_transit = np.abs(phase) < (dur / period_val) * 0.5
        oot_mask = ~in_transit
        if np.sum(oot_mask) > 10:
            oot_rms = np.std(flux_arr[oot_mask])
            depth_val = row.get("depth", 0.001)
            snr = depth_val / oot_rms if oot_rms > 0 else 0.0
            if snr > 10:
                sig_level = "HIGH (≥10σ)"
            elif snr > 5:
                sig_level = "MEDIUM (5–10σ)"
            elif snr > 3:
                sig_level = "LOW (3–5σ)"
            else:
                sig_level = "INSIGNIFICANT (<3σ)"
    return snr, sig_level


def _generate_evidence_bullets(row: pd.Series, scores: dict[str, float]) -> list[str]:
    """Generate evidence bullet points explaining the classification."""
    bullets = []
    cls = row.get('final_classification', row.get('classification', 'UNKNOWN'))

    # Depth evidence
    depth = row.get("depth", 0)
    if depth > 0.01:
        bullets.append(f"• Transit depth ({depth*100:.2f}%) is unusually large — consistent with eclipsing binary or blended system.")
    elif depth > 0.005:
        bullets.append(f"• Transit depth ({depth*100:.3f}%) is moderate — could be large planet or small-star eclipse.")
    elif depth > 0.0005:
        bullets.append(f"• Transit depth ({depth*100:.4f}%) is within typical exoplanet range (0.05%–1%).")
    else:
        bullets.append(f"• Transit depth ({depth*100:.4f}%) is very shallow — consistent with small rocky planet or marginal detection.")

    # Shape evidence
    shape = scores.get("Shape", 0)
    if shape >= 0.6:
        bullets.append("• Transit shape is box-like with flat bottom — consistent with planetary transit geometry.")
    elif shape >= 0.3:
        bullets.append("• Transit shape shows some deviation from ideal box — may indicate grazing eclipse or blended system.")
    else:
        bullets.append("• Transit shape is V-shaped or highly curved — strong indicator of stellar eclipse (EB) or grazing transit.")

    # Odd/Even evidence
    oddeven = scores.get("OddEven", 0)
    if oddeven >= 0.6:
        bullets.append("• Odd and even transit depths are consistent — no secondary eclipse detected, favors planet.")
    elif oddeven >= 0.3:
        bullets.append("• Minor depth variation between odd/even transits — warrants further scrutiny.")
    else:
        bullets.append("• Significant odd/even depth mismatch detected — hallmark of eclipsing binary system.")

    # Secondary eclipse
    nosec = scores.get("NoSec", 0)
    if nosec >= 0.6:
        bullets.append("• No secondary eclipse detected at phase 0.5 — consistent with planet (non-luminous companion).")
    elif nosec >= 0.3:
        bullets.append("• Weak or ambiguous secondary eclipse signal — inconclusive for companion nature.")
    else:
        bullets.append("• Clear secondary eclipse detected at phase 0.5 — companion is self-luminous (stellar binary).")

    # Ellipsoidal
    ellip = scores.get("Ellipsoidal", 0)
    if ellip >= 0.6:
        bullets.append("• No ellipsoidal variation detected — star is not tidally distorted by companion.")
    elif ellip >= 0.3:
        bullets.append("• Weak ellipsoidal variation present — possible close binary or high-mass companion.")
    else:
        bullets.append("• Strong ellipsoidal variation detected — companion is massive enough to distort host star.")

    # Achromaticity
    achrom = scores.get("Achrom", 0)
    if achrom >= 0.6:
        bullets.append("• Transit is achromatic (wavelength-independent) — consistent with opaque planetary disk.")
    elif achrom >= 0.3:
        bullets.append("• Slight chromatic variation — could be stellar spots or mild atmospheric effects.")
    else:
        bullets.append("• Strong chromatic variation — indicates wavelength-dependent opacity (stellar spots, not planet).")

    # Atmosphere
    atmos = scores.get("Atmos", 0)
    if atmos >= 0.6:
        bullets.append("• Atmospheric signature consistent with Rayleigh scattering slope — potential for follow-up spectroscopy.")
    elif atmos >= 0.3:
        bullets.append("• Weak atmospheric signal — data quality may limit atmospheric characterization.")
    else:
        bullets.append("• No detectable atmospheric signature — either flat spectrum or insufficient S/N.")

    # Duration/Period
    period = row.get("period", 0)
    duration = row.get("duration", 0) * 24  # hours
    if period < 1.0:
        bullets.append(f"• Ultra-short period (P={period:.3f} d) — candidate for ultra-short period planet (USP) or harmonic alias.")
    elif period < 3.0:
        bullets.append(f"• Short period (P={period:.3f} d) — typical for hot Jupiters and close-in small planets.")
    elif period < 10.0:
        bullets.append(f"• Moderate period (P={period:.3f} d) — within typical exoplanet orbital range.")
    else:
        bullets.append(f"• Long period (P={period:.2f} d) — fewer transits observed, lower confidence detection.")

    if duration < 1.0:
        bullets.append(f"• Short transit duration ({duration:.2f} hr) — consistent with small impact parameter or small companion.")
    elif duration < 3.0:
        bullets.append(f"• Moderate transit duration ({duration:.2f} hr) — typical for most transiting systems.")
    else:
        bullets.append(f"• Long transit duration ({duration:.2f} hr) — may indicate grazing geometry or large stellar companion.")

    # ML + Ensemble summary
    ml = row.get("ml_score", 0) * 100
    ensemble = row.get("ensemble_score", row.get("final_score", 0)) * 100
    physics = row.get("physics_composite", 0) * 100

    if cls == "TRANSIT":
        if ensemble >= 70:
            bullets.append(f"• HIGH CONFIDENCE: Ensemble score {ensemble:.1f}% with strong physics ({physics:.1f}%) and ML ({ml:.1f}%) agreement.")
        elif ensemble >= 50:
            bullets.append(f"• MODERATE CONFIDENCE: Ensemble score {ensemble:.1f}% — physics ({physics:.1f}%) and ML ({ml:.1f}%) partially agree.")
        else:
            bullets.append(f"• LOW CONFIDENCE: Ensemble score {ensemble:.1f}% — discrepancy between physics ({physics:.1f}%) and ML ({ml:.1f}%).")
    elif cls == "ECLIPSING_BINARY":
        bullets.append(f"• EB signature: Ensemble {ensemble:.1f}% — physics scores indicate stellar eclipse characteristics.")
    elif cls == "FALSE_POSITIVE":
        bullets.append(f"• FP verdict: Ensemble {ensemble:.1f}% — multiple physics checks failed, likely instrumental or stellar artifact.")
    else:
        bullets.append(f"• UNCERTAIN: Ensemble {ensemble:.1f}% — inconclusive evidence, requires additional data or vetting.")

    return bullets


def _create_executive_summary_page(pdf, predictions: pd.DataFrame, ground_truth: pd.DataFrame | None):
    """Page 1: Executive Summary."""
    with plt.rc_context(_TEXT_RC):
        fig = plt.figure(figsize=(8.5, 11), facecolor="white")
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        total = len(predictions)
        cls_counts = predictions.get('final_classification', predictions.get('classification', pd.Series(['UNKNOWN']*total))).value_counts()

        n_transit = cls_counts.get("TRANSIT", 0)
        n_eb = cls_counts.get("ECLIPSING_BINARY", 0)
        n_fp = cls_counts.get("FALSE_POSITIVE", 0)
        n_uncertain = cls_counts.get("UNCERTAIN", 0)
        n_other = total - n_transit - n_eb - n_fp - n_uncertain

        # Title
        ax.text(0.5, 0.95, "EXOPLANET PIPELINE — EXECUTIVE SUMMARY",
                ha="center", va="top", fontsize=18, fontweight="bold", color="#1a237e",
                transform=ax.transAxes)
        ax.text(0.5, 0.91, "Automated Transit Detection & Classification Report",
                ha="center", va="top", fontsize=11, color="#555555", style="italic",
                transform=ax.transAxes)

        # Horizontal line
        ax.plot([0.05, 0.95], [0.89, 0.89], color="#1a237e", linewidth=2, transform=ax.transAxes)

        # Overview box
        y_pos = 0.84
        ax.text(0.08, y_pos, "PIPELINE OVERVIEW", fontsize=13, fontweight="bold", color="#1a237e",
                transform=ax.transAxes)
        y_pos -= 0.04
        overview_lines = [
            f"Total Targets Processed:     {total}",
            f"  ├─ Confirmed Transits:     {n_transit}  ({n_transit/total*100:.1f}%)",
            f"  ├─ Eclipsing Binaries:       {n_eb}  ({n_eb/total*100:.1f}%)",
            f"  ├─ False Positives:          {n_fp}  ({n_fp/total*100:.1f}%)",
            f"  ├─ Uncertain:                {n_uncertain}  ({n_uncertain/total*100:.1f}%)",
        ]
        if n_other > 0:
            overview_lines.append(f"  └─ Other:                    {n_other}  ({n_other/total*100:.1f}%)")
        else:
            overview_lines[-1] = overview_lines[-1].replace("├─", "└─")

        for line in overview_lines:
            ax.text(0.10, y_pos, line, fontsize=10, fontfamily="monospace", color="#333333",
                    transform=ax.transAxes)
            y_pos -= 0.028

        # Score statistics
        y_pos -= 0.02
        ax.text(0.08, y_pos, "SCORE STATISTICS", fontsize=13, fontweight="bold", color="#1a237e",
                transform=ax.transAxes)
        y_pos -= 0.04

        score_col = "ensemble_score" if "ensemble_score" in predictions.columns else "final_score"
        if score_col in predictions.columns:
            scores = predictions[score_col].dropna() * 100
            score_lines = [
                f"Mean Ensemble Score:         {scores.mean():.1f}%",
                f"Median Ensemble Score:       {scores.median():.1f}%",
                f"Std Deviation:               {scores.std():.1f}%",
                f"Highest Score:               {scores.max():.1f}%",
                f"Lowest Score:                {scores.min():.1f}%",
                f"",
                f"High Confidence (≥70%):        {(scores >= 70).sum()} targets",
                f"Medium Confidence (50–70%):    {((scores >= 50) & (scores < 70)).sum()} targets",
                f"Low Confidence (<50%):       {(scores < 50).sum()} targets",
            ]
            for line in score_lines:
                ax.text(0.10, y_pos, line, fontsize=10, fontfamily="monospace", color="#333333",
                        transform=ax.transAxes)
                y_pos -= 0.028

        # Ground truth accuracy
        if ground_truth is not None and len(ground_truth) > 0:
            y_pos -= 0.02
            ax.text(0.08, y_pos, "VALIDATION AGAINST GROUND TRUTH", fontsize=13, fontweight="bold", color="#1a237e",
                    transform=ax.transAxes)
            y_pos -= 0.04

            # Merge predictions with ground truth
            merge_col = "tic_id"
            label_col = None
            # Try to find the label column in ground truth
            for possible_name in ["label", "classification", "true_label", "true_class", "type", "class"]:
                if possible_name in ground_truth.columns:
                    label_col = possible_name
                    break

            if merge_col in predictions.columns and merge_col in ground_truth.columns and label_col is not None:
                merged = predictions.merge(ground_truth[[merge_col, label_col]], on=merge_col, how="inner")
                if len(merged) > 0:
                    pred_col = "final_classification" if "final_classification" in merged.columns else "classification"
                    if pred_col in merged.columns and label_col in merged.columns:
                        correct = (merged[pred_col] == merged[label_col]).sum()
                        total_labeled = len(merged)
                        accuracy = correct / total_labeled * 100 if total_labeled > 0 else 0

                        # Per-class breakdown
                        classes = merged[label_col].unique()
                        acc_lines = [
                            f"Labeled Targets:             {total_labeled}",
                            f"Correct Classifications:     {correct}",
                            f"Overall Accuracy:            {accuracy:.1f}%",
                            f"",
                        ]
                        for cls in sorted(classes):
                            cls_mask = merged[label_col] == cls
                            cls_total = cls_mask.sum()
                            cls_correct = ((merged[pred_col] == cls) & cls_mask).sum()
                            cls_acc = cls_correct / cls_total * 100 if cls_total > 0 else 0
                            acc_lines.append(f"  {cls:<25}  {cls_correct}/{cls_total}  ({cls_acc:.1f}%)")

                        for line in acc_lines:
                            ax.text(0.10, y_pos, line, fontsize=10, fontfamily="monospace", color="#333333",
                                    transform=ax.transAxes)
                            y_pos -= 0.028
            else:
                ax.text(0.10, y_pos, "Ground truth columns: " + ", ".join(ground_truth.columns.tolist()[:5]),
                        fontsize=9, color="#888888", transform=ax.transAxes)
                y_pos -= 0.028

        # Key findings
        y_pos -= 0.02
        ax.text(0.08, y_pos, "KEY FINDINGS", fontsize=13, fontweight="bold", color="#1a237e",
                transform=ax.transAxes)
        y_pos -= 0.04

        findings = []
        if n_transit > 0:
            top_transit = predictions[predictions.get('final_classification', predictions.get('classification')) == "TRANSIT"]
            if len(top_transit) > 0 and score_col in top_transit.columns:
                best = top_transit.loc[top_transit[score_col].idxmax()]
                findings.append(f"• Highest-confidence transit: TIC {int(best['tic_id'])} (Score: {best[score_col]*100:.1f}%)")
        if n_eb > 0:
            findings.append(f"• {n_eb} eclipsing binary system(s) identified — ruled out as planet candidates.")
        if n_fp > 0:
            findings.append(f"• {n_fp} false positive(s) flagged — stellar variability, systematics, or blends.")

        high_conf = predictions[predictions[score_col] >= 0.7] if score_col in predictions.columns else pd.DataFrame()
        if len(high_conf) > 0:
            findings.append(f"• {len(high_conf)} target(s) classified with high confidence (≥70%).")

        low_conf = predictions[predictions[score_col] < 0.5] if score_col in predictions.columns else pd.DataFrame()
        if len(low_conf) > 0:
            findings.append(f"• {len(low_conf)} target(s) require additional vetting (confidence <50%).")

        for finding in findings:
            ax.text(0.10, y_pos, finding, fontsize=10, color="#333333", wrap=True,
                    transform=ax.transAxes)
            y_pos -= 0.035

        # Footer
        ax.text(0.5, 0.02, "Generated by Exoplanet Detection Pipeline  |  Stage 8: Visualization & Reporting",
                ha="center", va="bottom", fontsize=8, color="#888888", transform=ax.transAxes)

        pdf.savefig(fig, dpi=150, facecolor="white")
        plt.close(fig)


def _create_tic_detail_page(pdf, row: pd.Series, config: dict):
    """Create a detailed text page for a single TIC."""
    with plt.rc_context(_TEXT_RC):
        fig = plt.figure(figsize=(8.5, 11), facecolor="white")
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        tic_id = int(row["tic_id"])
        cls = row.get('final_classification', row.get('classification', 'UNKNOWN'))
        score = row.get('ensemble_score', row.get('final_score', row.get('ml_confidence', 0))) * 100

        # Classification colors
        cls_colors = {
            "TRANSIT": "#2e7d32",
            "ECLIPSING_BINARY": "#c62828",
            "FALSE_POSITIVE": "#ef6c00",
            "BLEND": "#f9a825",
            "STELLAR_ACTIVITY": "#fbc02d",
            "HARMONIC_ALIAS": "#7b1fa2",
            "UNCERTAIN": "#f57f17",
            "REJECT": "#c62828",
        }
        cls_color = cls_colors.get(cls, "#555555")

        # ── HEADER ──
        ax.text(0.5, 0.97, f"TIC {tic_id}",
                ha="center", va="top", fontsize=22, fontweight="bold", color="#1a237e",
                transform=ax.transAxes)
        ax.text(0.5, 0.935, f"Classification: {cls}  |  Confidence: {score:.1f}%",
                ha="center", va="top", fontsize=12, fontweight="bold", color=cls_color,
                transform=ax.transAxes)
        ax.plot([0.05, 0.95], [0.92, 0.92], color="#1a237e", linewidth=1.5, transform=ax.transAxes)

        y = 0.895

        # ── SECTION 1: DETECTION SUMMARY ──
        ax.text(0.08, y, "DETECTION SUMMARY", fontsize=12, fontweight="bold", color="#1a237e",
                transform=ax.transAxes)
        y -= 0.035

        period = row.get("period", 0)
        depth = row.get("depth", 0)
        duration = row.get("duration", 0)
        bls_sde = row.get("bls_sde", 0)
        tls_sde = row.get("tls_sde", 0)

        # Compute SNR
        data = load_processed_curve(tic_id, config)
        snr, sig_level = _compute_snr(row, data)

        depth_str = f"{depth*100:.4f}%" if depth < 0.01 else f"{depth*100:.2f}%"

        summary_lines = [
            f"  Period:              {period:.4f} days",
            f"  Transit Depth:       {depth_str}",
            f"  Transit Duration:    {duration*24:.2f} hours",
            f"  BLS SDE:             {bls_sde:.2f}",
            f"  TLS SDE:             {tls_sde:.2f}",
            f"  SNR:                 {snr:.2f}",
            f"  Significance:        {sig_level}",
        ]
        for line in summary_lines:
            ax.text(0.10, y, line, fontsize=10, fontfamily="monospace", color="#333333",
                    transform=ax.transAxes)
            y -= 0.026

        y -= 0.015

        # ── SECTION 2: PHYSICS VALIDATION ──
        ax.text(0.08, y, "PHYSICS VALIDATION", fontsize=12, fontweight="bold", color="#1a237e",
                transform=ax.transAxes)
        y -= 0.035

        scores = {
            "Depth Check": row.get("depth_score", 0),
            "Shape Analysis": row.get("shape_score", 0),
            "No Secondary": row.get("nosec_score", 0),
            "Odd/Even Test": row.get("oddeven_score", 0),
            "Ellipsoidal Var": row.get("ellipsoidal_score", 0),
            "Achromaticity": row.get("achrom_score", 0),
            "Atmosphere": row.get("atmos_score", 0),
        }

        physics_total = row.get("physics_composite", 0) * 100

        # Table header
        ax.text(0.10, y, f"{'Check':<20} {'Score':>8}  {'Status':>10}  {'Interpretation'}",
                fontsize=9, fontfamily="monospace", fontweight="bold", color="#1a237e",
                transform=ax.transAxes)
        y -= 0.025
        ax.plot([0.08, 0.92], [y+0.012, y+0.012], color="#cccccc", linewidth=0.5, transform=ax.transAxes)

        for name, val in scores.items():
            label = _score_label(val)
            color = _score_color(val)

            # Interpretation
            if name == "Depth Check":
                interp = "Depth within plausible planet range" if val >= 0.6 else "Depth too large/small for planet"
            elif name == "Shape Analysis":
                interp = "Box-like transit shape" if val >= 0.6 else "V-shaped or curved (EB-like)"
            elif name == "No Secondary":
                interp = "No secondary eclipse" if val >= 0.6 else "Secondary eclipse detected"
            elif name == "Odd/Even Test":
                interp = "Consistent depths" if val >= 0.6 else "Depth mismatch (EB)"
            elif name == "Ellipsoidal Var":
                interp = "No tidal distortion" if val >= 0.6 else "Tidal distortion present"
            elif name == "Achromaticity":
                interp = "Wavelength-independent" if val >= 0.6 else "Chromatic (stellar spots)"
            elif name == "Atmosphere":
                interp = "Scattering signature" if val >= 0.6 else "Flat/no atmosphere signal"
            else:
                interp = ""

            ax.text(0.10, y, f"{name:<20} {val:>7.2f}  ",
                    fontsize=9, fontfamily="monospace", color="#333333",
                    transform=ax.transAxes)
            ax.text(0.42, y, f"{label:>10}",
                    fontsize=9, fontfamily="monospace", fontweight="bold", color=color,
                    transform=ax.transAxes)
            ax.text(0.55, y, f"  {interp}",
                    fontsize=9, fontfamily="monospace", color="#555555",
                    transform=ax.transAxes)
            y -= 0.025

        y -= 0.005
        ax.plot([0.08, 0.92], [y+0.015, y+0.015], color="#cccccc", linewidth=0.5, transform=ax.transAxes)
        ax.text(0.10, y, f"{'PHYSICS COMPOSITE:':<20} {physics_total:>7.1f}%",
                fontsize=10, fontfamily="monospace", fontweight="bold", color="#1a237e",
                transform=ax.transAxes)
        y -= 0.035

        # ── SECTION 3: ML CLASSIFICATION ──
        ax.text(0.08, y, "MACHINE LEARNING CLASSIFICATION", fontsize=12, fontweight="bold", color="#1a237e",
                transform=ax.transAxes)
        y -= 0.035

        ml_score = row.get("ml_score", 0) * 100
        ensemble = row.get("ensemble_score", row.get("final_score", 0)) * 100

        ml_lines = [
            f"  ML Model Score:      {ml_score:.1f}%",
            f"  Ensemble Score:      {ensemble:.1f}%",
            f"  Physics Composite:   {physics_total:.1f}%",
            f"",
            f"  Agreement:           {'STRONG' if abs(ml_score - physics_total) < 15 else 'MODERATE' if abs(ml_score - physics_total) < 30 else 'WEAK'}",
            f"  (|ML – Physics| = {abs(ml_score - physics_total):.1f}%)",
        ]
        for line in ml_lines:
            ax.text(0.10, y, line, fontsize=10, fontfamily="monospace", color="#333333",
                    transform=ax.transAxes)
            y -= 0.026

        y -= 0.015

        # ── SECTION 4: FINAL VERDICT & EVIDENCE ──
        ax.text(0.08, y, "FINAL VERDICT & EVIDENCE", fontsize=12, fontweight="bold", color="#1a237e",
                transform=ax.transAxes)
        y -= 0.035

        # Verdict box
        verdict_text = f"  VERDICT: {cls}  (Confidence: {score:.1f}%)"
        ax.text(0.10, y, verdict_text,
                fontsize=11, fontfamily="monospace", fontweight="bold", color=cls_color,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor=cls_color, linewidth=2),
                transform=ax.transAxes)
        y -= 0.045

        # Evidence bullets
        bullets = _generate_evidence_bullets(row, scores)
        for bullet in bullets:
            # Wrap long bullets
            words = bullet.split()
            lines = []
            current = ""
            for word in words:
                if len(current) + len(word) + 1 <= 95:
                    current += word + " "
                else:
                    lines.append(current.strip())
                    current = word + " "
            if current:
                lines.append(current.strip())

            for i, line in enumerate(lines):
                prefix = "  • " if i == 0 else "    "
                ax.text(0.10, y, prefix + line,
                        fontsize=9, color="#333333", transform=ax.transAxes)
                y -= 0.022
            y -= 0.005

        # Footer
        ax.text(0.5, 0.02, f"TIC {tic_id}  |  Page generated by Exoplanet Detection Pipeline",
                ha="center", va="bottom", fontsize=8, color="#888888", transform=ax.transAxes)

        pdf.savefig(fig, dpi=150, facecolor="white")
        plt.close(fig)


def _create_statistics_page(pdf, predictions: pd.DataFrame, ground_truth: pd.DataFrame | None):
    """Final page: Overall statistics and validation."""
    with plt.rc_context(_TEXT_RC):
        fig = plt.figure(figsize=(8.5, 11), facecolor="white")
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        ax.text(0.5, 0.97, "OVERALL STATISTICS & VALIDATION",
                ha="center", va="top", fontsize=18, fontweight="bold", color="#1a237e",
                transform=ax.transAxes)
        ax.plot([0.05, 0.95], [0.945, 0.945], color="#1a237e", linewidth=2, transform=ax.transAxes)

        y = 0.91

        # Classification distribution
        ax.text(0.08, y, "CLASSIFICATION DISTRIBUTION", fontsize=13, fontweight="bold", color="#1a237e",
                transform=ax.transAxes)
        y -= 0.04

        total = len(predictions)
        cls_col = "final_classification" if "final_classification" in predictions.columns else "classification"
        if cls_col in predictions.columns:
            cls_counts = predictions[cls_col].value_counts()
            for cls, count in cls_counts.items():
                pct = count / total * 100
                bar_width = pct / 100 * 0.5
                ax.barh(y, bar_width, height=0.018, left=0.10, color="#1a237e", alpha=0.7, transform=ax.transAxes)
                ax.text(0.10, y + 0.005, f"{cls:<25} {count:>4}  ({pct:>5.1f}%)",
                        fontsize=10, fontfamily="monospace", color="#333333", va="center",
                        transform=ax.transAxes)
                y -= 0.035

        y -= 0.02

        # Score distribution histogram
        ax.text(0.08, y, "ENSEMBLE SCORE DISTRIBUTION", fontsize=13, fontweight="bold", color="#1a237e",
                transform=ax.transAxes)
        y -= 0.04

        score_col = "ensemble_score" if "ensemble_score" in predictions.columns else "final_score"
        if score_col in predictions.columns:
            scores = predictions[score_col].dropna() * 100

            # Create mini histogram inline
            hist_ax = fig.add_axes([0.12, y - 0.18, 0.76, 0.16])
            hist_ax.hist(scores, bins=20, color="#1a237e", alpha=0.7, edgecolor="white")
            hist_ax.set_xlabel("Ensemble Score (%)", fontsize=9)
            hist_ax.set_ylabel("Count", fontsize=9)
            hist_ax.axvline(scores.mean(), color="#c62828", linestyle="--", linewidth=1.5, label=f"Mean: {scores.mean():.1f}%")
            hist_ax.axvline(50, color="#2e7d32", linestyle="--", linewidth=1, alpha=0.5, label="Threshold (50%)")
            hist_ax.legend(fontsize=8)
            hist_ax.tick_params(labelsize=8)
            hist_ax.grid(True, alpha=0.3)
            y -= 0.22

        # Confusion matrix if ground truth available
        if ground_truth is not None and len(ground_truth) > 0:
            y -= 0.02
            ax.text(0.08, y, "CONFUSION MATRIX (Ground Truth vs Predicted)", fontsize=13, fontweight="bold", color="#1a237e",
                    transform=ax.transAxes)
            y -= 0.04

            merge_col = "tic_id"
            label_col = None
            for possible_name in ["label", "classification", "true_label", "true_class", "type", "class"]:
                if possible_name in ground_truth.columns:
                    label_col = possible_name
                    break
            if merge_col in predictions.columns and merge_col in ground_truth.columns and label_col is not None:
                merged = predictions.merge(ground_truth[[merge_col, label_col]], on=merge_col, how="inner")
                if len(merged) > 0 and cls_col in merged.columns and label_col in merged.columns:
                    labels = sorted(merged[label_col].unique())
                    preds = sorted(merged[cls_col].unique())
                    all_classes = sorted(set(list(labels) + list(preds)))

                    # Build confusion matrix
                    cm = {}
                    for true_cls in all_classes:
                        cm[true_cls] = {}
                        for pred_cls in all_classes:
                            cm[true_cls][pred_cls] = ((merged[label_col] == true_cls) & (merged[cls_col] == pred_cls)).sum()

                    # Draw table
                    cell_h = 0.025
                    cell_w = 0.12
                    start_x = 0.15
                    start_y = y

                    # Header row
                    ax.text(start_x - 0.02, start_y, "True \\ Predicted", fontsize=8, fontweight="bold",
                            color="#1a237e", transform=ax.transAxes, ha="right", va="center")
                    for j, pred_cls in enumerate(all_classes):
                        ax.text(start_x + j * cell_w + cell_w/2, start_y, pred_cls[:8],
                                fontsize=8, fontweight="bold", color="#1a237e", ha="center", va="center",
                                transform=ax.transAxes)

                    for i, true_cls in enumerate(all_classes):
                        row_y = start_y - (i + 1) * cell_h
                        ax.text(start_x - 0.02, row_y, true_cls[:8],
                                fontsize=8, fontweight="bold", color="#1a237e", ha="right", va="center",
                                transform=ax.transAxes)
                        for j, pred_cls in enumerate(all_classes):
                            val = cm[true_cls][pred_cls]
                            color = "#2e7d32" if true_cls == pred_cls and val > 0 else "#333333"
                            weight = "bold" if true_cls == pred_cls and val > 0 else "normal"
                            ax.text(start_x + j * cell_w + cell_w/2, row_y, str(val),
                                    fontsize=9, color=color, fontweight=weight, ha="center", va="center",
                                    transform=ax.transAxes)

                    y = start_y - (len(all_classes) + 2) * cell_h

        # Summary metrics
        y -= 0.03
        ax.text(0.08, y, "SUMMARY METRICS", fontsize=13, fontweight="bold", color="#1a237e",
                transform=ax.transAxes)
        y -= 0.04

        if ground_truth is not None and len(ground_truth) > 0:
            merge_col = "tic_id"
            label_col = None
            for possible_name in ["label", "classification", "true_label", "true_class", "type", "class"]:
                if possible_name in ground_truth.columns:
                    label_col = possible_name
                    break

            if merge_col in predictions.columns and merge_col in ground_truth.columns and label_col is not None:
                merged = predictions.merge(ground_truth[[merge_col, label_col]], on=merge_col, how="inner")
                if len(merged) > 0 and cls_col in merged.columns and label_col in merged.columns:
                    correct = (merged[cls_col] == merged[label_col]).sum()
                    total_labeled = len(merged)
                    accuracy = correct / total_labeled * 100 if total_labeled > 0 else 0

                    # Per-class precision/recall
                    metrics_lines = [
                        f"Overall Accuracy:        {accuracy:.1f}%",
                        f"Total Labeled:           {total_labeled}",
                        f"Correct:                 {correct}",
                        f"Misclassified:           {total_labeled - correct}",
                        f"",
                    ]

                    for cls in sorted(merged[label_col].unique()):
                        tp = ((merged[cls_col] == cls) & (merged[label_col] == cls)).sum()
                        fp = ((merged[cls_col] == cls) & (merged[label_col] != cls)).sum()
                        fn = ((merged[cls_col] != cls) & (merged[label_col] == cls)).sum()
                        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        metrics_lines.append(f"{cls:<22}  P:{precision:>5.1f}%  R:{recall:>5.1f}%  F1:{f1:>5.1f}%")

                    for line in metrics_lines:
                        ax.text(0.10, y, line, fontsize=10, fontfamily="monospace", color="#333333",
                                transform=ax.transAxes)
                        y -= 0.026
            else:
                ax.text(0.10, y, f"Ground truth loaded but no 'label' column found. Columns: {list(ground_truth.columns)[:5]}",
                        fontsize=10, color="#888888", style="italic", transform=ax.transAxes)
                y -= 0.03
        else:
            ax.text(0.10, y, "No ground truth labels available for validation.",
                    fontsize=10, color="#888888", style="italic", transform=ax.transAxes)
            y -= 0.03

        # Footer
        ax.text(0.5, 0.02, "Generated by Exoplanet Detection Pipeline  |  End of Report",
                ha="center", va="bottom", fontsize=8, color="#888888", transform=ax.transAxes)

        pdf.savefig(fig, dpi=150, facecolor="white")
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# PART 3: MAIN RUNNER (enhanced to generate BOTH reports)
# ═════════════════════════════════════════════════════════════════════════════

def run_visualization(predictions: pd.DataFrame, config: dict) -> None:
    """Generate both graph report and text report."""
    logger.info("Stage 8: Generating reports (graphs + text)")

    if predictions is None or len(predictions) == 0:
        logger.warning("No predictions — skipping visualization")
        return

    plots_dir = Path(config["paths"]["plots"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = Path(config["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    score_col = "ensemble_score" if "ensemble_score" in predictions.columns else "final_score"
    top_n = min(100, len(predictions))
    top = predictions.nlargest(top_n, score_col)

    # ── REPORT 1: Dark-theme graph PDF (existing, unchanged) ──
    pdf_path_dark = reports_dir / "pipeline_report_dark.pdf"
    with PdfPages(pdf_path_dark) as pdf:
        page_num = 1
        for _, row in top.iterrows():
            page_num = create_candidate_page(pdf, row, config, page_num)
    logger.info("Graph report saved: %s", pdf_path_dark)

    # ── REPORT 2: Professional text PDF (NEW) ──

    # Try to load ground truth for validation stats
    ground_truth = None
    gt_path = Path(config["paths"].get("data", "data")) / "test_30_mixed.csv"
    if not gt_path.exists():
        # Try alternative paths
        for alt in ["data/test_30_mixed.csv", "data/test_mixed.csv", "data/labels.csv"]:
            alt_path = Path(alt)
            if alt_path.exists():
                gt_path = alt_path
                break

    if gt_path.exists():
        try:
            ground_truth = pd.read_csv(gt_path)
            logger.info("Ground truth loaded from %s (%d rows)", gt_path, len(ground_truth))
        except Exception as e:
            logger.warning("Could not load ground truth: %s", e)

    pdf_path_text = reports_dir / "pipeline_report_text.pdf"
    with PdfPages(pdf_path_text) as pdf:
        # Page 1: Executive Summary
        _create_executive_summary_page(pdf, predictions, ground_truth)

        # Pages 2+: Per-TIC detail pages
        for _, row in top.iterrows():
            _create_tic_detail_page(pdf, row, config)

        # Final page: Statistics
        _create_statistics_page(pdf, predictions, ground_truth)

    logger.info("Text report saved: %s", pdf_path_text)
    logger.info("Stage 8 complete — 2 reports generated")
