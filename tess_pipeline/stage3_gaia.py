"""Stage 3: Stellar parameter cross-match — GOD MODE with SIMBAD fallback."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astroquery.simbad import Simbad

from tess_pipeline.utils import expected_earth_depth, expected_jupiter_depth

logger = logging.getLogger(__name__)

TIC_CATALOG_PATH = Path("data/tic_catalog.csv")
TIC_MINIMAL_PATH = Path("data/tic_8_1_minimal.csv.gz")

# In-memory cache for TIC catalog
_TIC_CATALOG = None


def _load_tic_catalog() -> pd.DataFrame | None:
    """Load TIC catalog from local file. Try minimal first, then full."""
    global _TIC_CATALOG
    if _TIC_CATALOG is not None:
        return _TIC_CATALOG

    for path in [TIC_MINIMAL_PATH, TIC_CATALOG_PATH]:
        if path.exists():
            try:
                logger.info("Loading TIC catalog from %s", path)
                if str(path).endswith(".gz"):
                    df = pd.read_csv(path, compression="gzip", low_memory=False)
                else:
                    df = pd.read_csv(path, low_memory=False)
                # Standardize column names
                df.columns = [c.upper().strip() for c in df.columns]
                _TIC_CATALOG = df
                logger.info("Loaded TIC catalog with %d rows", len(df))
                return df
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)
    
    return None


def _query_simbad_for_tic(tic_id: int) -> dict | None:
    """Query SIMBAD for TIC object to get stellar parameters."""
    try:
        Simbad.TIMEOUT = 30
        result = Simbad.query_object(f"TIC {tic_id}")
        if result is not None and len(result) > 0:
            row = result[0]
            teff = None
            if 'Teff' in result.colnames:
                teff = float(row['Teff']) if pd.notna(row['Teff']) else None
            elif 'Fe_H_Teff' in result.colnames:
                teff = float(row['Fe_H_Teff']) if pd.notna(row['Fe_H_Teff']) else None
            
            # Estimate radius from spectral type if available
            r_star = 1.0
            if 'SP_TYPE' in result.colnames and row['SP_TYPE']:
                sp = str(row['SP_TYPE'])
                if sp.startswith('O'): r_star = 15.0
                elif sp.startswith('B'): r_star = 7.0
                elif sp.startswith('A'): r_star = 2.5
                elif sp.startswith('F'): r_star = 1.4
                elif sp.startswith('G'): r_star = 1.0
                elif sp.startswith('K'): r_star = 0.7
                elif sp.startswith('M'): r_star = 0.3
            
            return {
                "teff": teff if teff else 5778.0,
                "r_star": r_star,
                "logg": 4.44,
                "mass": r_star,  # Rough approximation
                "bp_rp": 0.82,
                "query_source": "simbad",
            }
    except Exception as e:
        logger.debug("SIMBAD query failed for TIC %s: %s", tic_id, e)
    
    return None


def _sun_like_defaults(tic_id: int) -> dict:
    """Return conservative Sun-like defaults when all queries fail."""
    return {
        "tic_id": tic_id,
        "source_id": None,
        "teff": 5778.0,
        "logg": 4.44,
        "r_star": 1.0,
        "mass": 1.0,
        "bp_rp": 0.82,
        "expected_earth_depth": expected_earth_depth(1.0),
        "expected_jupiter_depth": expected_jupiter_depth(1.0),
        "query_status": "FALLBACK",
        "query_source": "sun-like_default",
    }


def query_stellar_params(tic_id: int, config: dict) -> dict:
    """Query stellar parameters from local TIC catalog. Falls back to SIMBAD, then defaults."""
    cache_dir = Path(config["paths"]["gaia_cache"])
    cache_file = cache_dir / f"{tic_id}.json"

    # Check cache first
    if config["gaia"]["cache_enabled"] and cache_file.exists():
        try:
            import json
            with open(cache_file) as f:
                cached = json.load(f)
            if cached.get("query_source") not in ["sun-like_default", None]:
                logger.info("Cache hit for TIC %s", tic_id)
                return cached
        except Exception:
            pass

    # Try local TIC catalog
    tic_df = _load_tic_catalog()
    if tic_df is not None and "ID" in tic_df.columns:
        row = tic_df[tic_df["ID"] == tic_id]
        if len(row) > 0:
            r = row.iloc[0]
            r_star = float(r.get("RADIUS", 1.0)) if pd.notna(r.get("RADIUS")) else 1.0
            teff = float(r.get("TEFF", 5778.0)) if pd.notna(r.get("TEFF")) else 5778.0
            logg = float(r.get("LOGG", 4.44)) if pd.notna(r.get("LOGG")) else 4.44
            mass = float(r.get("MASS", 1.0)) if pd.notna(r.get("MASS")) else 1.0
            bp_rp = float(r.get("BP-RP", 0.82)) if pd.notna(r.get("BP-RP")) else 0.82

            result = {
                "tic_id": tic_id,
                "source_id": int(r.get("GAIA3ID", 0)) if pd.notna(r.get("GAIA3ID")) else None,
                "teff": teff,
                "logg": logg,
                "r_star": r_star,
                "mass": mass,
                "bp_rp": bp_rp,
                "expected_earth_depth": expected_earth_depth(r_star),
                "expected_jupiter_depth": expected_jupiter_depth(r_star),
                "query_status": "SUCCESS",
                "query_source": "tic_catalog_local",
            }
            
            if config["gaia"]["cache_enabled"]:
                import json
                with open(cache_file, "w") as f:
                    json.dump(result, f, indent=2)
            return result

    # Try SIMBAD fallback
    simbad_result = _query_simbad_for_tic(tic_id)
    if simbad_result is not None:
        result = {
            "tic_id": tic_id,
            "source_id": None,
            "teff": simbad_result["teff"],
            "logg": simbad_result["logg"],
            "r_star": simbad_result["r_star"],
            "mass": simbad_result["mass"],
            "bp_rp": simbad_result["bp_rp"],
            "expected_earth_depth": expected_earth_depth(simbad_result["r_star"]),
            "expected_jupiter_depth": expected_jupiter_depth(simbad_result["r_star"]),
            "query_status": "SUCCESS",
            "query_source": "simbad",
        }
        if config["gaia"]["cache_enabled"]:
            import json
            with open(cache_file, "w") as f:
                json.dump(result, f, indent=2)
        return result

    # Fallback to Sun-like defaults
    print(f"  ⚠️  No catalog/SIMBAD data for TIC {tic_id}, using Sun-like defaults")
    result = _sun_like_defaults(tic_id)
    
    if config["gaia"]["cache_enabled"]:
        import json
        with open(cache_file, "w") as f:
            json.dump(result, f, indent=2)
    
    return result


def run_gaia_crossmatch(candidates: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Query stellar params for all candidate TIC IDs from local catalog."""
    logger.info("Stage 3: Stellar parameter lookup for %d candidates", len(candidates))
    
    if candidates is None or len(candidates) == 0:
        logger.warning("Stage 3: No candidates to process")
        return pd.DataFrame()

    print(f"\n{'='*60}")
    print(f"  STAGE 3: STELLAR PARAMETER LOOKUP")
    print(f"{'='*60}")
    
    stellar_rows = []
    tic_ids = candidates["tic_id"].unique().tolist()
    
    for tic_id in tic_ids:
        stellar_rows.append(query_stellar_params(int(tic_id), config))
        # Rate limit SIMBAD queries
        time.sleep(0.5)

    stellar_df = pd.DataFrame(stellar_rows)
    merged = candidates.merge(stellar_df, on="tic_id", how="left")

    # Summary
    n_success = (stellar_df["query_source"] == "tic_catalog_local").sum()
    n_simbad = (stellar_df["query_source"] == "simbad").sum()
    n_fallback = (stellar_df["query_source"] == "sun-like_default").sum()
    
    print(f"\n  TIC Catalog: {n_success} sources")
    print(f"  SIMBAD:      {n_simbad} sources")
    print(f"  Fallback:    {n_fallback} sources")
    print(f"{'='*60}\n")

    out_path = Path(config["paths"]["gaia_cache"]) / "stellar_params.csv"
    stellar_df.to_csv(out_path, index=False)
    logger.info("Stage 3 complete: stellar params saved to %s", out_path)
    return merged