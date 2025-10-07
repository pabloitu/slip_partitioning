from datetime import datetime
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from cat_handler import paths

# ---- unified output schema (no T/N/P axes; keep error columns for consistency) ----
FIELDS: List[str] = [
    "id", "time_iso", "longitude", "latitude", "depth", "mag", "mag_type",
    "lon_error", "lat_error", "depth_error", "mag_error",
    "strike1", "dip1", "rake1", "strike2", "dip2", "rake2",
    "Mrr", "Mtt", "Mpp", "Mrt", "Mrp", "Mtp",
    "source"
]

# ---- default spatial/magnitude filter (same region you’ve been using elsewhere) ----
LAT_MIN, LAT_MAX = -51.0, -31.0
LON_MIN, LON_MAX = -80.0, -65.0
MAG_MIN = 4.8  # NaN magnitudes are KEPT


def _to_float(x: Any) -> Optional[float]:
    try:
        f = float(x)
        return f if np.isfinite(f) else np.nan
    except Exception:
        return np.nan


def _iso_from_parts(y: int, m: int, d: int, hh: int, mm: int, sec_float: float) -> str:
    si = int(sec_float)
    micros = int(round((sec_float - si) * 1_000_000))
    if micros >= 1_000_000:
        si += 1
        micros -= 1_000_000
    if si >= 60:
        si -= 60
        mm += 1
        if mm >= 60:
            mm -= 60
            hh += 1
    dt = datetime(int(y), int(m), int(d), int(hh), int(mm), int(si), micros)
    return dt.isoformat(timespec="milliseconds" if micros else "seconds")


def _make_id(y: int, m: int, d: int, hh: int, mm: int, sec_float: float, idx: int) -> str:
    si = int(sec_float)
    ms = int(round((sec_float - si) * 1000.0))
    return f"potin{y:04d}{m:02d}{d:02d}{hh:02d}{mm:02d}{si:02d}{ms:03d}_{idx:06d}"


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in FIELDS:
        if c not in df.columns:
            df[c] = np.nan
    return df


def parse_potin(in_path: str,
                out_path: str,
                lon_min: float = LON_MIN,
                lon_max: float = LON_MAX,
                lat_min: float = LAT_MIN,
                lat_max: float = LAT_MAX,
                mag_min: float = MAG_MIN) -> None:
    # load input
    df = pd.read_csv(in_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {"year", "month", "day", "hour", "minute", "seconds", "longitude", "latitude", "depth"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in POTIN CSV: {missing}")

    # numeric coercion
    for c in ("year", "month", "day", "hour", "minute"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["seconds"]   = pd.to_numeric(df["seconds"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"]  = pd.to_numeric(df["latitude"], errors="coerce")
    df["depth"]     = pd.to_numeric(df["depth"], errors="coerce")
    df["magnitude"] = pd.to_numeric(df.get("magnitude", np.nan), errors="coerce")
    if "magnitude_type" not in df.columns:
        df["magnitude_type"] = np.nan

    # build unified rows
    rows: List[dict] = []
    for idx, r in df.reset_index(drop=True).iterrows():
        # time
        y  = int(r["year"])   if pd.notna(r["year"])   else 1970
        mo = int(r["month"])  if pd.notna(r["month"])  else 1
        d  = int(r["day"])    if pd.notna(r["day"])    else 1
        hh = int(r["hour"])   if pd.notna(r["hour"])   else 0
        mm = int(r["minute"]) if pd.notna(r["minute"]) else 0
        ss = float(r["seconds"]) if pd.notna(r["seconds"]) else 0.0
        time_iso = _iso_from_parts(y, mo, d, hh, mm, ss)
        eid = _make_id(y, mo, d, hh, mm, ss, idx)

        # magnitude
        mag = _to_float(r.get("magnitude"))
        mtype_raw = r.get("magnitude_type")
        mag_type = None
        if isinstance(mtype_raw, str):
            mt = mtype_raw.strip().lower()
            mag_type = mt if mt and mt != "nan" else None

        rows.append({
            "id": eid,
            "time_iso": time_iso,
            "longitude": _to_float(r.get("longitude")),
            "latitude":  _to_float(r.get("latitude")),
            "depth":     _to_float(r.get("depth")),
            "mag":       mag,
            "mag_type":  mag_type,
            "lon_error":   np.nan,
            "lat_error":   np.nan,
            "depth_error": np.nan,
            "mag_error":   np.nan,
            "strike1": np.nan, "dip1": np.nan, "rake1": np.nan,
            "strike2": np.nan, "dip2": np.nan, "rake2": np.nan,
            "Mrr": np.nan, "Mtt": np.nan, "Mpp": np.nan, "Mrt": np.nan, "Mrp": np.nan, "Mtp": np.nan,
            "source": "potin2025",
        })

    out = pd.DataFrame(rows)
    out = _ensure_columns(out)

    # ---- filters: lon/lat window and minimum magnitude (keep NaN magnitudes) ----
    mask = (
        (out["longitude"].between(lon_min, lon_max, inclusive="both")) &
        (out["latitude"].between(lat_min, lat_max, inclusive="both"))
    )
    if mag_min is not None:
        mask = mask & (out["mag"].ge(mag_min) | out["mag"].isna())

    out = out.loc[mask].reset_index(drop=True)

    # write
    out.to_csv(out_path, index=False)
    print(f"[POTIN] wrote {out_path} with {len(out)} rows (filtered by "
          f"lon[{lon_min},{lon_max}], lat[{lat_min},{lat_max}], mag≥{mag_min} keeping NaNs).")


def main() -> None:
    parse_potin(str(paths.raw_potin), str(paths.cat_potin),
                lon_min=LON_MIN, lon_max=LON_MAX,
                lat_min=LAT_MIN, lat_max=LAT_MAX,
                mag_min=MAG_MIN)


if __name__ == "__main__":
    main()
