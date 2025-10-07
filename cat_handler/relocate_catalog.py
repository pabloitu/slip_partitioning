

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from cat_handler import paths

# ---------------- Config ----------------
TIME_TOL_S: float = 60.0   # time matching window (seconds)
MAG_TOL: float   = 0.8      # magnitude tolerance (different scales)
DEFAULT_DEPTH = 33.0

# ---------------- Schema ----------------
FIELDS: List[str] = [
    "id", "time_iso", "longitude", "latitude", "depth", "mag", "mag_type",
    "lon_error", "lat_error", "depth_error", "mag_error",
    "strike1", "dip1", "rake1", "strike2", "dip2", "rake2",
    "Mrr", "Mtt", "Mpp", "Mrt", "Mrp", "Mtp",
    "source", "dups",
]

# -------------- Helpers -----------------
def _finite(x: Any) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def _to_epoch(series: pd.Series) -> np.ndarray:
    """Return seconds since epoch (float), NaN if invalid."""
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    out = np.full(len(series), np.nan, dtype="float64")
    mask = ts.notna()
    if mask.any():
        try:
            out[mask.to_numpy()] = (ts[mask].view("int64") // 10**9).astype(float)
        except Exception:
            out[mask.to_numpy()] = (ts[mask].astype("int64") // 10**9).astype(float)
    return out

def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance (km). Returns inf if any input non-finite."""
    if not all(_finite(v) for v in (lon1, lat1, lon2, lat2)):
        return float("inf")
    R = 6371.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl   = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def _ensure_error_object_dtype(df: pd.DataFrame) -> None:
    """Allow writing the string 'relocated' in error columns."""
    for c in ("lon_error", "lat_error", "depth_error"):
        if c in df.columns:
            df[c] = df[c].astype(object)

def _ensure_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure all unified fields exist (unused set to NaN)."""
    for c in FIELDS:
        if c not in df.columns:
            df[c] = np.nan
    return df

# -------------- Matching core --------------
def _best_match_indices(
    t0: float,
    m0: float | None,
    lon0: float | None,
    lat0: float | None,
    pot_epochs: np.ndarray,
    pot_mags: np.ndarray,
    pot_lons: np.ndarray,
    pot_lats: np.ndarray,
    time_tol_s: float,
    mag_tol: float
) -> Optional[int]:
    """
    Find best POTIN index for a single merged event based on:
    1) |Δt| <= time_tol_s
    2) |ΔM| <= mag_tol OR potin_mag is NaN OR merged_mag is NaN
    Choose by min |Δt|, then min |ΔM| (NaN treated as +inf), then min distance.
    Returns index or None if no candidate.
    """
    if not np.isfinite(t0):
        return None

    # Time window via searchsorted
    left = np.searchsorted(pot_epochs, t0 - time_tol_s, side="left")
    right = np.searchsorted(pot_epochs, t0 + time_tol_s, side="right")
    if left >= right:
        return None

    cand_idxs = np.arange(left, right)

    # Magnitude tolerance
    pm = pot_mags[cand_idxs]
    if m0 is None or not np.isfinite(m0):
        mag_ok = np.ones_like(pm, dtype=bool)  # accept all if merged mag unknown
        dmag = np.full_like(pm, np.inf, dtype=float)
    else:
        with np.errstate(invalid="ignore"):
            dmag = np.abs(pm - m0)
        mag_ok = np.isnan(pm) | (dmag <= mag_tol)

    if not mag_ok.any():
        return None

    cand_idxs = cand_idxs[mag_ok]
    if cand_idxs.size == 0:
        return None

    # Ranking within candidates
    dt = np.abs(pot_epochs[cand_idxs] - t0)

    # For tie-breaks: use |ΔM| with NaN treated as +inf
    dmag_c = dmag[mag_ok]
    dmag_c = np.where(np.isfinite(dmag_c), dmag_c, np.inf)

    # Final tie-break: distance
    if lon0 is None or lat0 is None or not all(_finite(v) for v in (lon0, lat0)):
        dist = np.full_like(dt, np.inf, dtype=float)
    else:
        lon_c = pot_lons[cand_idxs]
        lat_c = pot_lats[cand_idxs]
        dist = np.array([haversine_km(lon0, lat0, lo, la) for lo, la in zip(lon_c, lat_c)], dtype=float)

    # rank by (|Δt|, |ΔM|, distance)
    order = np.lexsort((dist, dmag_c, dt))
    return int(cand_idxs[order[0]])

# -------------- Main relocation --------------
def relocate_with_potin(
    merged_csv: str,
    potin_csv: str,
    out_csv: str,
    time_tol_s: float = TIME_TOL_S,
    mag_tol: float = MAG_TOL,
    default_depth: float = DEFAULT_DEPTH
) -> None:
    # Load
    merged = pd.read_csv(merged_csv)
    potin  = pd.read_csv(potin_csv)

    merged = _ensure_fields(merged)
    potin  = _ensure_fields(potin)

    # Allow string in error fields
    _ensure_error_object_dtype(merged)

    # Precompute epochs & arrays for POTIN (sorted by time)
    potin_epochs = _to_epoch(potin["time_iso"])
    order = np.argsort(np.where(np.isfinite(potin_epochs), potin_epochs, np.inf))
    potin = potin.iloc[order].reset_index(drop=True)
    potin_epochs = potin_epochs[order].astype(float)

    potin_mags = pd.to_numeric(potin["mag"], errors="coerce").to_numpy(dtype=float)
    potin_lons = pd.to_numeric(potin["longitude"], errors="coerce").to_numpy(dtype=float)
    potin_lats = pd.to_numeric(potin["latitude"],  errors="coerce").to_numpy(dtype=float)
    potin_deps = pd.to_numeric(potin["depth"],     errors="coerce").to_numpy(dtype=float)

    # Prepare merged vectors
    merged_epochs = _to_epoch(merged["time_iso"])
    merged_mags   = pd.to_numeric(merged["mag"], errors="coerce").to_numpy(dtype=float)
    merged_lons   = pd.to_numeric(merged["longitude"], errors="coerce").to_numpy(dtype=float)
    merged_lats   = pd.to_numeric(merged["latitude"],  errors="coerce").to_numpy(dtype=float)
    merged_deps   = pd.to_numeric(merged["depth"],  errors="coerce").to_numpy(dtype=float)

    # Iterate and update where matched
    lon_new  = merged["longitude"].to_numpy(dtype=object)
    lat_new  = merged["latitude"].to_numpy(dtype=object)
    dep_new  = merged["depth"].to_numpy(dtype=object)

    lon_err  = merged["lon_error"].to_numpy(dtype=object)
    lat_err  = merged["lat_error"].to_numpy(dtype=object)
    dep_err  = merged["depth_error"].to_numpy(dtype=object)

    for i in range(len(merged)):
        t0   = merged_epochs[i]
        m0   = merged_mags[i] if np.isfinite(merged_mags[i]) else None
        lon0 = merged_lons[i] if np.isfinite(merged_lons[i]) else None
        lat0 = merged_lats[i] if np.isfinite(merged_lats[i]) else None

        j = _best_match_indices(
            t0, m0, lon0, lat0,
            potin_epochs, potin_mags, potin_lons, potin_lats,
            time_tol_s=time_tol_s, mag_tol=mag_tol
        )
        if j is None:
            if merged_deps[i] == 30.:
                dep_err[i] = "default"
            continue

        # Update coordinates to relocated
        lon_new[i] = potin_lons[j]
        lat_new[i] = potin_lats[j]
        dep_new[i] = potin_deps[j]

        # Mark errors as 'relocated'
        lon_err[i] = "relocated"
        lat_err[i] = "relocated"
        dep_err[i] = "relocated"

    merged["longitude"]  = lon_new
    merged["latitude"]   = lat_new
    merged["depth"]      = dep_new
    merged["lon_error"]  = lon_err
    merged["lat_error"]  = lat_err
    merged["depth_error"]= dep_err

    # Save
    merged.to_csv(out_csv, index=False)
    print(f"[RELOC] wrote {out_csv}  ({len(merged)} rows)  "
          f"(time_tol_s={time_tol_s}, mag_tol={mag_tol})")

def main() -> None:
    relocate_with_potin(
        merged_csv=str(paths.cat_merged),
        potin_csv=str(paths.cat_potin),
        out_csv=str(paths.cat_relocated),
        time_tol_s=TIME_TOL_S,
        mag_tol=MAG_TOL,
        default_depth=33,
    )

if __name__ == "__main__":
    main()
