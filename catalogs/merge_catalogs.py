from __future__ import annotations

import os
import re
from math import radians, sin, cos, asin, sqrt
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# -------------------- OUTPUT SCHEMA --------------------
FIELDS = [
    "id","time_iso","longitude","latitude","depth","mag","mag_type",
    "strike1","dip1","rake1","strike2","dip2","rake2",
    "T_plunge","T_azimuth","N_plunge","N_azimuth","P_plunge","P_azimuth",
    "Mrr","Mtt","Mpp","Mrt","Mrp","Mtp","source","dups"
]

def _finite(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def ensure_fields(df: pd.DataFrame) -> pd.DataFrame:
    needed = set(FIELDS) - {"dups"}
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan
    if "source" not in df.columns:
        df["source"] = np.nan
    return df

def load_catalog_generic(path: str, source_hint: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = ensure_fields(df)
    df["source"] = df["source"].astype(str).str.strip().str.lower()
    if source_hint:
        df["source"] = df["source"].where(
            (df["source"] != "") & (df["source"] != "nan"),
            other=source_hint.strip().lower()
        )
    df["time_iso"] = df["time_iso"].astype(str)
    return df

# Canonicalize ID: lower, strip; remove a SINGLE leading letter if next char is a digit.
_LETTER_DIGIT = re.compile(r'^[A-Za-z](\d.*)$')
def canonical_id(s: Any) -> str:
    s = (str(s or "")).strip().lower()
    m = _LETTER_DIGIT.match(s)
    return m.group(1) if m else s

def haversine_km(lon1, lat1, lon2, lat2):
    if not all(_finite(v) for v in (lon1, lat1, lon2, lat2)):
        return np.inf
    lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2.0)**2 + cos(lat1)*cos(lat2)*sin(dlon/2.0)**2
    return 2.0 * 6371.0 * asin(sqrt(a))

def build_canon_map() -> Dict[str,str]:
    return {
        "gcmt": "gcmt", "cmt": "gcmt",
        "us": "us", "usgs": "us", "anss": "us", "iscgem": "iscgem",
        "gfz": "gfz", "geofon": "gfz", "neic": "neic", "neis": "neis",
        "ipgp": "ipgp", "sja": "sja", "isc": "isc", "gem": "gem",
        "gmt": "gmt",
    }

def normalize_source(row: Dict[str, Any], canon_map: Dict[str,str], file_hint: Optional[str] = None) -> str:
    src = (row.get("source") or "").strip().lower()
    if not src and file_hint:
        src = file_hint.strip().lower()
    if not src:
        return "other"
    return canon_map.get(src, src)

def rank_source(src: str, priority: Dict[str,int]) -> int:
    return priority.get((src or "other").lower(), priority.get("other", 999))

# -------------------- DSU --------------------
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

# -------------------- De-dup: R1 + R2 --------------------
def dedup_clusters(records: List[Dict[str, Any]],
                   dt_near_s: float,
                   km_near: float,
                   dmag_near: float) -> List[List[int]]:
    n = len(records)
    dsu = DSU(n)

    # R1: same canonical id
    cids = [canonical_id(r.get("id")) for r in records]
    by_id: Dict[str, List[int]] = {}
    for i, cid in enumerate(cids):
        if cid:
            by_id.setdefault(cid, []).append(i)
    for idxs in by_id.values():
        for a, b in zip(idxs, idxs[1:]):
            dsu.union(a, b)

    # R2 inputs
    times = pd.to_datetime([r.get("time_iso") for r in records], utc=True, errors="coerce")
    valid_time = times.notna()
    tsec = pd.Series(np.full(n, np.nan), dtype="float64")
    tsec.loc[valid_time] = (times[valid_time].astype("int64") // 10**9).astype(float)

    lons = np.array([r.get("longitude") for r in records], dtype=object)
    lats = np.array([r.get("latitude")  for r in records], dtype=object)
    mags = np.array([r.get("mag")       for r in records], dtype=object)

    idx_valid = np.where(valid_time)[0]
    order = idx_valid[np.argsort(tsec.loc[idx_valid].to_numpy())]

    for pos_i, i in enumerate(order):
        ti = tsec.iloc[i]
        jpos = pos_i + 1
        while jpos < len(order):
            j = order[jpos]
            tj = tsec.iloc[j]
            dt = tj - ti
            if dt > dt_near_s:
                break
            if all(_finite(x) for x in (lons[i], lats[i], lons[j], lats[j], mags[i], mags[j])):
                dkm = haversine_km(lons[i], lats[i], lons[j], lats[j])
                dmag = abs(float(mags[i]) - float(mags[j]))
                if (dkm <= km_near) and (dmag <= dmag_near):
                    dsu.union(i, j)
            jpos += 1

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = dsu.find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())

# -------------------- Field-wise selection by source priority --------------------
def _choose_by_priority(records: List[Dict[str, Any]], idxs: List[int],
                        fields: Optional[List[str]],
                        require_all: bool,
                        canon_map: Dict[str,str],
                        priority: Dict[str,int]) -> Optional[Dict[str, Any]]:
    best = None; best_rank = 10**9
    for ii in idxs:
        r = records[ii]
        rk = rank_source(normalize_source(r, canon_map), priority)
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

def assemble_output_record(records: List[Dict[str, Any]], idxs: List[int],
                           canon_map: Dict[str,str],
                           priority: Dict[str,int]) -> Dict[str, Any]:
    glob = _choose_by_priority(records, idxs, fields=None, require_all=False,
                               canon_map=canon_map, priority=priority) or records[idxs[0]]

    loc   = _choose_by_priority(records, idxs, ["longitude","latitude","depth"], True,  canon_map, priority) or glob
    magr  = _choose_by_priority(records, idxs, ["mag","mag_type"],              True,  canon_map, priority) or glob
    ten   = _choose_by_priority(records, idxs, ["Mrr","Mtt","Mpp","Mrt","Mrp","Mtp"], True,  canon_map, priority)
    sdr   = _choose_by_priority(records, idxs, ["strike1","dip1","rake1","strike2","dip2","rake2"], True, canon_map, priority)
    axes  = _choose_by_priority(records, idxs, ["T_plunge","T_azimuth","N_plunge","N_azimuth","P_plunge","P_azimuth"], True, canon_map, priority)

    out: Dict[str, Any] = {}
    out["id"]       = glob.get("id")
    out["time_iso"] = glob.get("time_iso")
    out["source"]   = normalize_source(glob, canon_map)

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

    chosen_id = str(out.get("id", ""))
    dups = []
    for ii in idxs:
        rid = str(records[ii].get("id", ""))
        if rid == chosen_id:
            continue
        dups.append(f"{normalize_source(records[ii], canon_map)}:{rid}")
    out["dups"] = ";".join(dups)
    return out

# -------------------- Duplicate-ID labeling for FULL catalog --------------------
def assign_duplicate_ids(records: List[Dict[str, Any]],
                         clusters: List[List[int]],
                         canon_map: Dict[str,str]) -> List[str]:
    """
    For each cluster, assign a running integer starting at 1.
    Each event gets 'clusterIndex_source' (source canonicalized).
    If the same source appears multiple times within the cluster,
    suffix them with 'a','b','c' to keep uniqueness.
    """
    # sort clusters by earliest event time to make numbering stable
    def rec_time(idx):
        t = pd.to_datetime(records[idx].get("time_iso"), utc=True, errors="coerce")
        return t.value if pd.notna(t) else np.int64(-2**62)
    clusters_sorted = sorted(clusters, key=lambda cl: min(rec_time(i) for i in cl))

    dup_id_by_idx = [""] * len(records)
    for cluster_num, cl in enumerate(clusters_sorted, start=1):
        # count per source
        per_src: Dict[str, int] = {}
        for i in cl:
            src = normalize_source(records[i], canon_map) or "other"
            k = per_src.get(src, 0)
            suffix = src if k == 0 else f"{src}{chr(ord('a') + k)}"
            dup_id_by_idx[i] = f"{cluster_num}_{suffix}"
            per_src[src] = k + 1
    return dup_id_by_idx

# -------------------- Driver --------------------
def merge_and_label(input_files: Sequence[str],
                    out_merged_path: str,
                    out_full_path: str,
                    dt_near_s: float = 120.0,
                    km_near: float = 80.0,
                    dmag_near: float = 0.8,
                    priority: Optional[Dict[str,int]] = None,
                    source_hints: Optional[Dict[str,str]] = None) -> None:

    canon_map = build_canon_map()
    if priority is None:
        priority = {
            "gmt": 1,
            "us": 2,
            "gcmt": 3,
            "gfz": 4, "iscgem": 4, "neic": 4, "gem": 4,
            "neis": 5,
            "ipgp": 6, "sja": 6,
            "isc": 7,
            "other": 9,
        }

    def infer_hint(path: str) -> Optional[str]:
        if not source_hints:
            return None
        bn = os.path.basename(path).lower()
        for key, val in source_hints.items():
            if key.lower() in bn:
                return val
        return None

    frames = [load_catalog_generic(p, source_hint=infer_hint(p)) for p in input_files]
    all_df = pd.concat(frames, ignore_index=True, sort=False)
    all_df["source"] = all_df["source"].astype(str).str.strip().str.lower()
    records = all_df.to_dict("records")

    clusters = dedup_clusters(records, dt_near_s=dt_near_s, km_near=km_near, dmag_near=dmag_near)

    merged = [assemble_output_record(records, cl, canon_map, priority) for cl in clusters]
    df_merged = pd.DataFrame(merged)
    if not df_merged.empty:
        df_merged = df_merged.sort_values(
            by="time_iso",
            key=lambda s: pd.to_datetime(s, utc=True, errors="coerce"),
            na_position="last"
        )
        cols = [c for c in FIELDS if c in df_merged.columns]
        os.makedirs(os.path.dirname(out_merged_path) or ".", exist_ok=True)
        df_merged.to_csv(out_merged_path, index=False, columns=cols)

    # ------- full with duplicate_id (ALL rows)
    dup_ids = assign_duplicate_ids(records, clusters, canon_map)
    df_full = all_df.copy()
    df_full["source"] = df_full["source"].apply(lambda s: (canon_map.get(str(s).lower(), str(s).lower()) if pd.notna(s) else "other"))
    df_full["duplicate_id"] = dup_ids
    # sort by time
    df_full = df_full.sort_values(
        by="time_iso",
        key=lambda s: pd.to_datetime(s, utc=True, errors="coerce"),
        na_position="last"
    )
    os.makedirs(os.path.dirname(out_full_path) or ".", exist_ok=True)
    df_full.to_csv(out_full_path, index=False)

    print(f"merged -> {out_merged_path}  ({len(df_merged)} rows)")
    print(f"full+dups -> {out_full_path}  ({len(df_full)} rows)")

# -------------------- Main --------------------
def main():
    base = "processed_catalogs"
    files = [
        os.path.join(base, "gcmt_formatted.csv"),
        os.path.join(base, "anss_formatted.csv"),
        os.path.join(base, "isc_formatted.csv"),
        os.path.join(base, "isc_gem_formatted.csv"),
        os.path.join(base, "gmt_nico_formatted.csv"),
    ]
    out_merged = os.path.join(base, "merged_catalog.csv")
    out_full   = os.path.join(base, "full_catalog_with_dups.csv")

    # Dedup windows (tweak here)
    DT_NEAR_S = 120.0
    KM_NEAR   = 80.0
    DMAG_NEAR = 0.8

    PRIORITY = {
        "gmt": 1,
        "us": 2,
        "gcmt": 3,
        "gfz": 4, "iscgem": 4, "neic": 4, "gem": 4,
        "neis": 5,
        "ipgp": 6, "sja": 6,
        "isc": 7,
        "other": 9,
    }

    SOURCE_HINTS = {
        "gcmt": "gcmt",
        "anss": "us",
        "isc_gem": "gem",
        "isc_formatted": "isc",
        "gmt_nico": "gmt",
    }

    merge_and_label(
        input_files=files,
        out_merged_path=out_merged,
        out_full_path=out_full,
        dt_near_s=DT_NEAR_S,
        km_near=KM_NEAR,
        dmag_near=DMAG_NEAR,
        priority=PRIORITY,
        source_hints=SOURCE_HINTS,
    )

if __name__ == "__main__":
    main()
