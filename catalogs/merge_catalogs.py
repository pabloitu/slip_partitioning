#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge + de-duplicate catalogs with two rules:

R1 (strict): same canonical id -> merge
   canonical id = lowercased id with ONE leading letter removed if it's followed by a digit
   e.g., 'B010886A' -> '010886a', 'C201102051611A' -> '201102051611a'

R2 (proximity): |Δt| ≤ 120 s  AND  distance ≤ 80 km  AND  |Δmag| ≤ 0.8 -> merge
   (if coords or magnitudes are missing, R2 does NOT merge — avoids false positives)

For each duplicate set, select fields by source priority (best provider per field).
Outputs a single CSV with a 'dups' column listing merged duplicates as 'source:id'.
"""

import re
from math import radians, sin, cos, asin, sqrt
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# -------------------- FILE PATHS --------------------
GCMT_FILE = "global_2/merged_gcmt_mechanisms.csv"                    # GCMT (source column may be missing)
USGS_FILE = "global_2/anss_mechanisms.csv"                           # USGS/ANSS (source values like 'us','iscgem')
ISC_FILE  = "global_2/isc_mechanisms.txt"                            # ISC (source values like 'gfz','ipgp','neic','neis','sja')
GMT_FILE  = "global_2/GMT_1976_2025_consolidado_conSlab.csv"         # custom GMT file
OUT_MERGED = "merged_all_catalogs.csv"

# -------------------- DEDUP WINDOWS --------------------
DT_NEAR_S = 120.0  # seconds
KM_NEAR   = 80.0   # km
DMAG_NEAR = 0.8    # magnitude units

# -------------------- SOURCES & PRIORITY --------------------
CANON = {
    # GCMT / legacy
    "gcmt": "gcmt", "cmt": "gcmt",
    # USGS / ANSS variants
    "us": "us", "usgs": "us", "anss": "us", "iscgem": "iscgem",
    # ISC contributors
    "gfz": "gfz", "neic": "neic", "neis": "neis", "ipgp": "ipgp", "sja": "sja", "isc": "isc",
    # GMT
    "gmt": "gmt",
}
PRIORITY = {
    "gcmt": 3,
    "us": 2,
    "gmt": 1,
    "gfz": 4,
    "iscgem": 4,
    "neic": 4,
    "neis": 5,
    "ipgp": 6,
    "sja": 6,
    "isc": 7,
    "other": 9,
}

# -------------------- OUTPUT SCHEMA --------------------
FIELDS = [
    "id","time_iso","longitude","latitude","depth","mag","mag_type",
    "strike1","dip1","rake1","strike2","dip2","rake2",
    "T_plunge","T_azimuth","N_plunge","N_azimuth","P_plunge","P_azimuth",
    "Mrr","Mtt","Mpp","Mrt","Mrp","Mtp","source","dups"
]

# ======================================================
# Helpers
# ======================================================
def _finite(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def normalize_source(row: Dict[str, Any], file_hint: Optional[str] = None) -> str:
    src = (row.get("source") or "").strip().lower()
    if not src and file_hint:
        src = file_hint.strip().lower()
    return CANON.get(src, src if src else (file_hint or "other")).lower()

def rank_source(src: str) -> int:
    return PRIORITY.get((src or "other").lower(), PRIORITY["other"])

# Canonicalize ID: lower, strip; remove a SINGLE leading letter if next char is a digit.
_LETTER_DIGIT = re.compile(r'^[A-Za-z](\d.*)$')
def canonical_id(s: Any) -> str:
    s = (str(s or "")).strip().lower()
    m = _LETTER_DIGIT.match(s)
    return m.group(1) if m else s

def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance in km; returns inf if any coord missing."""
    if not all(_finite(v) for v in (lon1, lat1, lon2, lat2)):
        return np.inf
    lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2.0)**2 + cos(lat1)*cos(lat2)*sin(dlon/2.0)**2
    c = 2.0 * asin(sqrt(a))
    return 6371.0 * c  # km

def ensure_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all possible output fields exist (filled with NaN)."""
    needed = set(FIELDS) - {"dups"}
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan
    return df

# ======================================================
# Loaders
# ======================================================
def load_catalog_generic(path: str, source_hint: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = ensure_fields(df)
    if "source" not in df.columns:
        df["source"] = np.nan
    df["source"] = df["source"].astype(str).str.strip().str.lower()
    if source_hint:
        df["source"] = df["source"].where(
            (df["source"] != "") & (df["source"] != "nan"), other=source_hint.strip().lower()
        )
    # keep time_iso as string
    df["time_iso"] = df["time_iso"].astype(str)
    return df

_GMT_TIME_RE = re.compile(r"^(\d{12})[A-Za-z]$")  # YYYYMMDDhhmm + trailing letter

def _gmt_id_to_time_iso(gmt_id: str) -> Optional[str]:
    s = str(gmt_id or "").strip()
    m = _GMT_TIME_RE.match(s)
    if not m:
        return None
    digits = m.group(1)
    yyyy = int(digits[0:4]); mm = int(digits[4:6]); dd = int(digits[6:8])
    hh   = int(digits[8:10]); mi = int(digits[10:12])
    return f"{yyyy:04d}-{mm:02d}-{dd:02d}T{hh:02d}:{mi:02d}:00"

def load_gmt(path: str) -> pd.DataFrame:
    """
    GMT CSV columns:
      ID,Lat,Lon,Depth_km,Mw,Strike_1,Dip_1,Rake_1,P_trend,P_plunge,T_trend,T_plunge,
      Strike_2,Dip_2,Rake_2,Strike_2_c,Dip_2_cal,Rake_2_cal,Slab1,Depth2Slab
    """
    raw = pd.read_csv(path)
    out = pd.DataFrame({
        "id":        raw.get("ID"),
        "time_iso":  raw["ID"].apply(_gmt_id_to_time_iso),
        "latitude":  raw.get("Lat"),
        "longitude": raw.get("Lon"),
        "depth":     raw.get("Depth_km"),
        "mag":       raw.get("Mw"),
        "mag_type":  pd.Series(["mw"] * len(raw)),
        "strike1":   raw.get("Strike_1"),
        "dip1":      raw.get("Dip_1"),
        "rake1":     raw.get("Rake_1"),
        "strike2":   raw.get("Strike_2"),
        "dip2":      raw.get("Dip_2"),
        "rake2":     raw.get("Rake_2"),
        "T_plunge":  raw.get("T_plunge"),
        "T_azimuth": raw.get("T_trend"),
        "P_plunge":  raw.get("P_plunge"),
        "P_azimuth": raw.get("P_trend"),
        "N_plunge":  np.nan,
        "N_azimuth": np.nan,
        "Mrr": np.nan, "Mtt": np.nan, "Mpp": np.nan, "Mrt": np.nan, "Mrp": np.nan, "Mtp": np.nan,
        "source":    pd.Series(["gmt"] * len(raw)),
    })
    return ensure_fields(out)

# ======================================================
# Union–Find (Disjoint Set)
# ======================================================
class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

# ======================================================
# De-dup: R1 (same canonical id) + R2 (time+space+mag)
# ======================================================
def dedup_clusters(records: List[Dict[str, Any]]) -> List[List[int]]:
    n = len(records)
    dsu = DSU(n)

    # --- R1: same canonical id
    cids = [canonical_id(r.get("id")) for r in records]
    by_id: Dict[str, List[int]] = {}
    for i, cid in enumerate(cids):
        if cid:  # non-empty
            by_id.setdefault(cid, []).append(i)
    for idxs in by_id.values():
        for a, b in zip(idxs, idxs[1:]):
            dsu.union(a, b)

    # --- Precompute arrays for R2
    times = pd.to_datetime([r.get("time_iso") for r in records], utc=True, errors="coerce")
    valid_time = times.notna()
    # seconds since epoch for valid times
    tsec = pd.Series(np.full(n, np.nan))
    tsec.loc[valid_time] = (times[valid_time].astype("int64") // 10**9).astype(float)

    lons = np.array([r.get("longitude") for r in records], dtype=object)
    lats = np.array([r.get("latitude")  for r in records], dtype=object)
    mags = np.array([r.get("mag")       for r in records], dtype=object)

    # indices of valid times sorted by time
    idx_valid = np.where(valid_time)[0]
    order = idx_valid[np.argsort(tsec.loc[idx_valid].to_numpy())]

    # --- R2 sweep: within 120s AND within 80 km AND |Δmag| ≤ 0.8
    for pos_i, i in enumerate(order):
        ti = tsec.iloc[i]
        jpos = pos_i + 1
        while jpos < len(order):
            j = order[jpos]
            tj = tsec.iloc[j]
            dt = tj - ti  # >= 0

            if dt > DT_NEAR_S:
                break

            # require coords and magnitudes to be finite
            if all(_finite(x) for x in (lons[i], lats[i], lons[j], lats[j], mags[i], mags[j])):
                dkm = haversine_km(lons[i], lats[i], lons[j], lats[j])
                dmag = abs(float(mags[i]) - float(mags[j]))
                if (dkm <= KM_NEAR) and (dmag <= DMAG_NEAR):
                    dsu.union(i, j)

            jpos += 1

    # collect components
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = dsu.find(i)
        groups.setdefault(r, []).append(i)

    return list(groups.values())

# ======================================================
# Field-wise selection by source priority
# ======================================================
def _choose_by_priority(records: List[Dict[str, Any]], idxs: List[int],
                        fields: Optional[List[str]], require_all: bool) -> Optional[Dict[str, Any]]:
    best = None; best_rank = 10**9
    for ii in idxs:
        r = records[ii]
        rk = rank_source(normalize_source(r))
        ok = True
        if fields:
            for k in fields:
                v = r.get(k, None)
                if require_all:
                    if k in ("Mrr","Mtt","Mpp","Mrt","Mrp","Mtp"):
                        if not _finite(v): ok = False; break
                    else:
                        if v is None or (isinstance(v, str) and v.strip() == ""):
                            ok = False; break
                        if not isinstance(v, str) and not _finite(v):
                            ok = False; break
        if ok and rk < best_rank:
            best = r; best_rank = rk
    return best

def assemble_output_record(records: List[Dict[str, Any]], idxs: List[int]) -> Dict[str, Any]:
    # global "best" for id/time/source (does not force field completeness)
    glob = _choose_by_priority(records, idxs, fields=None, require_all=False) or records[idxs[0]]

    # per-field winners
    loc   = _choose_by_priority(records, idxs, ["longitude","latitude","depth"], require_all=True) or glob
    magr  = _choose_by_priority(records, idxs, ["mag","mag_type"], require_all=True) or glob
    ten   = _choose_by_priority(records, idxs, ["Mrr","Mtt","Mpp","Mrt","Mrp","Mtp"], require_all=True)
    sdr   = _choose_by_priority(records, idxs, ["strike1","dip1","rake1","strike2","dip2","rake2"], require_all=True)
    axes  = _choose_by_priority(records, idxs, ["T_plunge","T_azimuth","N_plunge","N_azimuth","P_plunge","P_azimuth"], require_all=True)

    out: Dict[str, Any] = {}
    out["id"]       = glob.get("id")
    out["time_iso"] = glob.get("time_iso")
    out["source"]   = normalize_source(glob)

    out["longitude"] = loc.get("longitude")
    out["latitude"]  = loc.get("latitude")
    out["depth"]     = loc.get("depth")

    out["mag"]      = magr.get("mag")
    mt = magr.get("mag_type")
    out["mag_type"] = (mt.lower() if isinstance(mt, str) else mt)

    for k in ("Mrr","Mtt","Mpp","Mrt","Mrp","Mtp"):
        out[k] = ten.get(k) if ten is not None else None

    for k in ("strike1","dip1","rake1","strike2","dip2","rake2"):
        out[k] = sdr.get(k) if sdr is not None else None

    for k in ("T_plunge","T_azimuth","N_plunge","N_azimuth","P_plunge","P_azimuth"):
        out[k] = axes.get(k) if axes is not None else None

    # dups list (exclude chosen id)
    chosen_id = str(out.get("id", ""))
    dups = []
    for ii in idxs:
        rid = str(records[ii].get("id", ""))
        if rid == chosen_id:
            continue
        dups.append(f"{normalize_source(records[ii])}:{rid}")
    out["dups"] = ";".join(dups)

    return out

# ======================================================
# Driver
# ======================================================
def merge_and_dedup(gcmt_path: str, usgs_path: str, isc_path: str, gmt_path: str, out_path: str):
    # Load with hints where needed
    df_gcmt = load_catalog_generic(gcmt_path, source_hint="gcmt")
    df_usgs = load_catalog_generic(usgs_path)   # per-row source present
    df_isc  = load_catalog_generic(isc_path)    # per-row source present
    df_gmt  = load_gmt(gmt_path)                # fixed 'gmt'

    all_df = pd.concat([df_gcmt, df_usgs, df_isc, df_gmt], ignore_index=True, sort=False)
    # normalize source column to canonical lower-case for all rows
    if "source" not in all_df.columns:
        all_df["source"] = np.nan
    all_df["source"] = all_df["source"].astype(str).str.strip().str.lower()

    records = all_df.to_dict("records")

    # build duplicate clusters using R1 + R2
    clusters = dedup_clusters(records)

    # assemble merged rows
    merged = [assemble_output_record(records, cl) for cl in clusters]
    out_df = pd.DataFrame(merged)

    # sort by time
    if not out_df.empty:
        out_df = out_df.sort_values(
            by="time_iso",
            key=lambda s: pd.to_datetime(s, utc=True, errors="coerce"),
            na_position="last"
        )
        cols = [c for c in FIELDS if c in out_df.columns]
        out_df.to_csv(out_path, index=False, columns=cols)

    print(f"Wrote {out_path} with {len(out_df)} events "
          f"(from {len(all_df)} inputs across {len(clusters)} clusters).")

# ------------------------------------------------------
if __name__ == "__main__":
    merge_and_dedup(GCMT_FILE, USGS_FILE, ISC_FILE, GMT_FILE, OUT_MERGED)
