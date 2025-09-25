# gcmt_ndk_parser_with_errors.py
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from cat_handler.parsers.tools import filter_df

# -------------------- CONFIG --------------------
LAT_MIN, LAT_MAX = -51.0, -27.0
LON_MIN, LON_MAX = -80.0, -65.0
MAG_MIN = 4.95
TIME_MIN = pd.to_datetime("1976-01-01T00:00:00", utc=True)

# -------------------- OUTPUT SCHEMA --------------------
# NOTE: T/N/P axes REMOVED. Error fields ADDED to match your ANSS schema.
FIELDS = [
    "id",
    "time_iso",
    "longitude",
    "latitude",
    "depth",
    "mag",
    "mag_type",
    "lon_error",
    "lat_error",
    "depth_error",
    "mag_error",
    "strike1",
    "dip1",
    "rake1",
    "strike2",
    "dip2",
    "rake2",
    "Mrr",
    "Mtt",
    "Mpp",
    "Mrt",
    "Mrp",
    "Mtp",
    "source",
]

# -------------------- helpers --------------------
def _fw(line: str, a1: int, b1: int) -> str:
    """Fixed-width slice using 1-based inclusive bounds per NDK spec."""
    s = line.ljust(80)  # NDK lines are 80 chars
    return s[a1 - 1 : b1].strip()

def _float_or_none(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _to_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()

def _parse_origin_datetime(date_s: str, time_s: str) -> Optional[datetime]:
    """
    date_s: 'YYYY/MM/DD'
    time_s: 'HH:MM:SS(.s)'
    """
    try:
        y, m, d = (int(x) for x in date_s.split("/"))
        hh, mm, rest = time_s.split(":")
        ss = float(rest)
        base = datetime(y, m, d, int(hh), int(mm), int(ss))
        frac = ss - int(ss)
        return base + timedelta(seconds=frac)
    except Exception:
        return None

def _float_tokens(s: str) -> List[float]:
    """Collect tokens convertible to float, in order (no regex)."""
    out: List[float] = []
    for tok in s.strip().split():
        try:
            out.append(float(tok))
        except Exception:
            continue
    return out

def _mw_from_scalar_moment(exp_dyncm: Optional[int], m0_scalar: Optional[float]) -> Optional[float]:
    """
    GCMT line 4 gives exponent 'exp' for dyne-cm; line 5 gives scalar (to be *10**exp dyne-cm).
    Mw = (2/3) * (log10(M0 [N*m]) - 9.1), with 1 N*m = 1e7 dyne-cm.
    """
    if exp_dyncm is None or m0_scalar is None or m0_scalar <= 0:
        return None
    # M0 in dyne-cm
    m0_dyncm = m0_scalar * (10.0 ** exp_dyncm)
    # convert to N*m
    m0_nm = m0_dyncm * 1e-7
    if m0_nm <= 0:
        return None
    return (2.0 / 3.0) * (np.log10(m0_nm) - 9.1)

def _moment_tensor_values(exp_dyncm: Optional[int], six_vals: List[Optional[float]]) -> List[Optional[float]]:
    """
    Convert six MT components from 'line-4 values' (unitless * 10**exp dyne-cm) to N*m.
    Output order: [Mrr, Mtt, Mpp, Mrt, Mrp, Mtp]
    """
    if exp_dyncm is None:
        return [None] * 6
    factor = 10.0 ** (exp_dyncm - 7)  # dyne-cm -> N*m
    out: List[Optional[float]] = []
    for v in six_vals:
        out.append(None if v is None else float(v) * factor)
    return out

# -------------------- NDK parsing (5-line blocks) --------------------
def _parse_block(l1: str, l2: str, l3: str, l4: str, l5: str) -> Optional[Dict[str, object]]:
    """
    Parse one 5-line NDK block into a row dict matching FIELDS.
    Avoids regex; uses fixed columns and float token collection.
    """
    # Line 1 (hypocenter reference)
    date_s = _fw(l1, 6, 15)       # YYYY/MM/DD
    time_s = _fw(l1, 17, 26)      # HH:MM:SS(.s)
    hdr_lat = _float_or_none(_fw(l1, 28, 33))
    hdr_lon = _float_or_none(_fw(l1, 35, 41))
    hdr_dep = _float_or_none(_fw(l1, 43, 47))

    origin_dt = _parse_origin_datetime(date_s, time_s)
    if origin_dt is None:
        return None

    # Line 2 (event id)
    ev_id = _fw(l2, 1, 16)

    # Line 3 (centroid params with 1-sigma errors, time shift first)
    # Format after 'CENTROID:' => t_shift t_err  lat lat_err  lon lon_err  depth depth_err
    try:
        cidx = l3.index("CENTROID:")
        tail = l3[cidx + len("CENTROID:") :]
    except ValueError:
        tail = ""
    f3 = _float_tokens(tail)
    # guard: require at least time shift + errors & lat/lon/depth + errors
    # Some entries may use fixed hypocenter; errors can be 0.0.
    tshift = f3[0] if len(f3) >= 1 else 0.0
    # t_err = f3[1] if len(f3) >= 2 else None  # not stored
    clat = f3[2] if len(f3) >= 3 else (hdr_lat if hdr_lat is not None else np.nan)
    clat_err = f3[3] if len(f3) >= 4 else np.nan
    clon = f3[4] if len(f3) >= 5 else (hdr_lon if hdr_lon is not None else np.nan)
    clon_err = f3[5] if len(f3) >= 6 else np.nan
    cdep = f3[6] if len(f3) >= 7 else (hdr_dep if hdr_dep is not None else np.nan)
    cdep_err = f3[7] if len(f3) >= 8 else np.nan

    centroid_dt = origin_dt + timedelta(seconds=float(tshift or 0.0))

    # Line 4 (exponent + six MT comps, each with its std err)
    f4 = _float_tokens(l4)
    exp_dyncm = int(f4[0]) if len(f4) >= 1 else None
    # positions: 1,3,5,7,9,11 are the six values (because it's value,error repeated)
    mt_vals = []
    for pos in (1, 3, 5, 7, 9, 11):
        mt_vals.append(f4[pos] if len(f4) > pos else None)
    Mrr, Mtt, Mpp, Mrt, Mrp, Mtp = _moment_tensor_values(exp_dyncm, mt_vals)

    # Line 5 (principal axes & scalar moment & nodal planes)
    f5 = _float_tokens(l5)
    # layout: [eT,plT,azT, eN,plN,azN, eP,plP,azP, scalar,  s1,d1,r1, s2,d2,r2]
    scalar = f5[9] if len(f5) >= 10 else None
    # nodal planes are the last 6 floats on the line
    if len(f5) >= 16:
        s1, d1, r1, s2, d2, r2 = f5[-6:]
    else:
        # fallback: if not enough tokens, leave None
        s1 = d1 = r1 = s2 = d2 = r2 = None

    # Magnitude: derive Mw from scalar moment if possible
    mw = np.round(_mw_from_scalar_moment(exp_dyncm, scalar), 1)
    mag_type = "mw" if mw is not None else None

    return {
        "id": ev_id,
        "time_iso": _to_iso(centroid_dt),
        "longitude": float(clon) if clon is not None else np.nan,
        "latitude": float(clat) if clat is not None else np.nan,
        "depth": float(cdep) if cdep is not None else np.nan,
        "mag": float(mw) if mw is not None else np.nan,
        "mag_type": mag_type,
        "lon_error": float(clon_err) if clon_err is not None else np.nan,
        "lat_error": float(clat_err) if clat_err is not None else np.nan,
        "depth_error": float(cdep_err) if cdep_err is not None else np.nan,
        "mag_error": np.nan,  # not provided in NDK
        "strike1": float(s1) if s1 is not None else np.nan,
        "dip1": float(d1) if d1 is not None else np.nan,
        "rake1": float(r1) if r1 is not None else np.nan,
        "strike2": float(s2) if s2 is not None else np.nan,
        "dip2": float(d2) if d2 is not None else np.nan,
        "rake2": float(r2) if r2 is not None else np.nan,
        "Mrr": Mrr,
        "Mtt": Mtt,
        "Mpp": Mpp,
        "Mrt": Mrt,
        "Mrp": Mrp,
        "Mtp": Mtp,
        "source": "gcmt",
    }

def parse_ndk_file(path: str | Path) -> List[Dict[str, object]]:
    """
    Read an NDK file and parse all 5-line event blocks.
    Ignores blank lines; requires multiples of 5 non-blank lines.
    """
    path = str(path)
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip() != ""]
    if len(lines) < 5:
        return rows
    # Truncate to multiple of 5
    lines = lines[: (len(lines) // 5) * 5]
    for i in range(0, len(lines), 5):
        rec = _parse_block(lines[i], lines[i + 1], lines[i + 2], lines[i + 3], lines[i + 4])
        if rec is not None:
            rows.append(rec)
    return rows

def prepare_gcmt(gcmt_1976_2020_path: str | Path,
                 gcmt_2021_2025_path: str | Path,
                 out_path: str | Path) -> pd.DataFrame:
    """
    Parse the two GCMT NDK catalogs, merge, filter (lon/lat/Mw/time), and write CSV.
    Filtering is applied to the merged dataframe (covers both files).
    """
    rows = []
    rows += parse_ndk_file(gcmt_1976_2020_path)
    rows += parse_ndk_file(gcmt_2021_2025_path)

    df = pd.DataFrame(rows, columns=FIELDS)
    # Drop rows with completely missing time or mag if desired; filter_df handles time_min/mag_min.
    df = filter_df(
        df,
        lon_min=LON_MIN,
        lon_max=LON_MAX,
        lat_min=LAT_MIN,
        lat_max=LAT_MAX,
        mag_min=MAG_MIN,
        time_min=TIME_MIN,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[GCMT] wrote {out_path} with {len(df)} rows.")
    return df

if __name__ == "__main__":
    from cat_handler import paths
    prepare_gcmt(paths.rawcat_gcmt_1976_2020, paths.rawcat_gcmt_2020_2025, paths.cat_gcmt)

