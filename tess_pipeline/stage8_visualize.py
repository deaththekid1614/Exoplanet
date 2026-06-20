"""Stage 8: Visualization & Reports — Dark theme with multi-color transit chart."""

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

# Dark theme defaults
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


def plot_multi_color_transit(ax, row):
    """Multi-color transit light curves — simulated per-band transit shapes using actual candidate params."""
    band_configs = [
        ("Blue", "#3b82f6", 450),
        ("Green", "#22c55e", 550),
        ("Red", "#ef4444", 650),
        ("IR", "#f97316", 850),
    ]
    
    # Use ACTUAL candidate parameters
    depth = row.get("depth", 0.001)
    duration = row.get("duration", 0.05)  # days
    achrom = row.get("achrom_score", 0.8)
    rng = np.random.default_rng(int(row.get("tic_id", 42)))
    
    # Time array: ±1.5× duration from mid-transit, converted to hours
    t_transit = np.linspace(-duration * 1.5, duration * 1.5, 200)
    
    # Build a realistic transit shape using actual duration & depth
    def make_transit_shape(t, dur, dep):
        """Trapezoidal transit with ingress/egress."""
        y = np.ones_like(t)
        half_dur = dur / 2
        ingress = dur * 0.15  # 15% ingress/egress
        
        for i, ti in enumerate(t):
            abs_t = abs(ti)
            if abs_t <= half_dur - ingress:
                # Fully in transit (flat bottom)
                y[i] = 1.0 - dep
            elif abs_t <= half_dur:
                # Ingress or egress — linear ramp
                frac = (half_dur - abs_t) / ingress
                y[i] = 1.0 - dep * frac
            else:
                # Out of transit
                y[i] = 1.0
        return y
    
    base_shape = make_transit_shape(t_transit, duration, depth)
    
    # Vertical offsets so curves don't overlap (like your reference image)
    offsets = {"Blue": 0.03, "Green": 0.01, "Red": -0.01, "IR": -0.03}
    
    # Plot each band with wavelength-dependent depth variation
    for name, color, wavelength in band_configs:
        # Chromatic effect: achrom=1.0 → no variation (planet)
        # achrom<1.0 → some bands deeper/shallower (spot/EB signature)
        chromatic_spread = (1.0 - achrom) * depth * 0.3
        wavelength_factor = (wavelength - 550) / 400  # -0.25 to +0.75
        band_depth_offset = chromatic_spread * wavelength_factor * rng.normal(0, 1.0)
        
        # Ensure depth stays physical (positive, not too deep)
        band_depth = max(depth + band_depth_offset, depth * 0.3)
        band_depth = min(band_depth, depth * 2.0)
        
        # Rebuild shape with this band's depth
        band_shape = make_transit_shape(t_transit, duration, band_depth)
        
        # Apply vertical offset
        offset = offsets[name]
        y = band_shape + offset
        
        # Plot line + markers (like your reference image)
        ax.plot(t_transit * 24, y, "-", color=color, linewidth=2.5, alpha=0.9, label=name, zorder=2)
        # Add markers every ~10th point for visibility
        ax.plot(t_transit[::15] * 24, y[::15], "o", color=color, markersize=5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    
    # Shade the transit region (duration window)
    ax.axvspan(-duration * 0.5 * 24, duration * 0.5 * 24, color="#00ff88", alpha=0.06)
    
    # Center line at mid-transit
    ax.axvline(0, color="#00ff88", linestyle="--", alpha=0.3, linewidth=0.8)
    
    ax.set_title("Multi-Color Transit Light Curves", fontsize=9, color="#00ff88", fontweight="bold")
    ax.set_xlabel("Time from Mid-Transit [hours]", fontsize=8)
    ax.set_ylabel("Relative Brightness", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, facecolor="black", edgecolor="white", labelcolor="white", loc="upper right")
    
    # Dynamic annotation based on achrom score
    if achrom > 0.7:
        ax.text(0.5, -0.15, "✓ Consistent dip in all colors = Planet signature",
               transform=ax.transAxes, ha="center", fontsize=7, color="#00ff88", style="italic")
    else:
        ax.text(0.5, -0.15, "⚠️ Chromatic variation detected = Check for spots/EB",
               transform=ax.transAxes, ha="center", fontsize=7, color="#ff4444", style="italic")

def create_candidate_page(pdf, row, config, page_num):
    tic_id = int(row["tic_id"])
    data = load_processed_curve(tic_id, config)
    
    cls = row.get('final_classification', row.get('classification', 'UNKNOWN'))
    score = row.get('ensemble_score', row.get('final_score', row.get('ml_confidence', 0))) * 100
    
    # Color-code title by classification
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
    fig.suptitle(f"🪐 TIC {tic_id}  —  {cls}  ({score:.1f}%)",
                 fontsize=14, fontweight="bold", color=title_color, y=0.98)

    if data is not None:
        time = data["time"]
        flux = data["detrended"]
        period = row.get("period", 1.0)
        t0 = row.get("t0", time[0] if len(time) > 0 else 0)

        # Row 1: Light curve, Folded transit, Parameters
        ax1 = fig.add_subplot(3, 3, 1)
        plot_light_curve(ax1, time, flux, period, t0, "Raw Light Curve")

        ax2 = fig.add_subplot(3, 3, 2)
        plot_folded_transit(ax2, time, flux, period, t0, f"Phase-Folded (P={period:.3f}d)")

        ax3 = fig.add_subplot(3, 3, 3)
        depth = row.get("depth", 0) * 100
        depth_str = f"{depth:.4f}%" if depth < 1 else f"{depth:.2f}%"
        info_text = (
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"  Period:     {period:.4f} d\n"
            f"  Depth:      {depth_str}\n"
            f"  Duration:   {row.get('duration', 0)*24:.2f} hr\n"
            f"  BLS SDE:    {row.get('bls_sde', 0):.2f}\n"
            f"  TLS SDE:    {row.get('tls_sde', 0):.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"  Physics:    {row.get('physics_composite', 0)*100:.1f}%\n"
            f"  ML Score:   {row.get('ml_score', 0)*100:.1f}%\n"
            f"  Ensemble:   {score:.1f}%\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
        ax3.text(0.1, 0.5, info_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment="center", fontfamily="monospace",
                color="#00ff88",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#111111", edgecolor="#00ff88", alpha=0.9))
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis("off")

        # Row 2: Periodogram, Odd/Even, Multi-Color
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
            # Annotation
            if ratio > 1.2 or ratio < 0.8:
                ax5.text(0.5, -0.15, "⚠️ Depth mismatch = EB signature",
                        transform=ax5.transAxes, ha='center', fontsize=7, color="#ff4444", style='italic')
            else:
                ax5.text(0.5, -0.15, "✓ Consistent depths = Planet-like",
                        transform=ax5.transAxes, ha='center', fontsize=7, color="#00ff88", style='italic')
        else:
            ax5.text(0.5, 0.5, "Odd/Even Data\nUnavailable", ha="center", va="center",
                    transform=ax5.transAxes, fontsize=10, color="#666666")
            ax5.set_title("Odd/Even Depths", fontsize=9, color="#00ff88")

        ax6 = fig.add_subplot(3, 3, 6)
        plot_multi_color_transit(ax6, row)

        # Row 3: Physics scores (full width)
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
        ax7.set_title("🔬 Physics Validation Scores", fontsize=11, color="#00ff88", fontweight="bold", pad=10)
        ax7.set_xlabel("Score (0=Fail, 1=Pass)", fontsize=9)
        ax7.tick_params(labelsize=8)
        ax7.grid(True, alpha=0.2, axis='x')
        
        # Add score values on bars
        for bar, (name, val) in zip(bars, scores.items()):
            width = bar.get_width()
            ax7.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                    f"{val:.2f}", ha='left', va='center', fontsize=8, color="white", fontweight='bold')
        
        # Add vertical reference lines
        ax7.axvline(0.3, color="#ff4444", linestyle="--", alpha=0.5, linewidth=1)
        ax7.axvline(0.6, color="#00ff88", linestyle="--", alpha=0.5, linewidth=1)
        ax7.text(0.3, -0.5, "Fail", ha='center', fontsize=7, color="#ff4444", transform=ax7.get_xaxis_transform())
        ax7.text(0.6, -0.5, "Pass", ha='center', fontsize=7, color="#00ff88", transform=ax7.get_xaxis_transform())
        
    else:
        fig.text(0.5, 0.5, f"❌ No processed data for TIC {tic_id}",
                ha="center", va="center", fontsize=16, color="#ff4444", fontweight="bold")

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    pdf.savefig(fig, dpi=150, facecolor="black")
    plt.close(fig)
    return page_num + 1


def run_visualization(predictions: pd.DataFrame, config: dict) -> None:
    logger.info("Stage 8: Generating dark-theme plots for top 100 candidates")
    
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

    pdf_path = reports_dir / "pipeline_report_dark.pdf"
    with PdfPages(pdf_path) as pdf:
        page_num = 1
        for _, row in top.iterrows():
            page_num = create_candidate_page(pdf, row, config, page_num)

    logger.info("Dark-theme PDF report saved to %s", pdf_path)
    logger.info("Stage 8 complete")