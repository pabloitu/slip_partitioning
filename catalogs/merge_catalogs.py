#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import re
import numpy as np
import pandas as pd

# ---- TUNABLE WINDOWS (temporal > spatial > magnitude) ----
TIME_WINDOW_S  = 60     # seconds
DIST_WINDOW_KM = 75     # km
MAG_WINDOW     = 0.5    # magnitude units
DUP_CRITERION  = "any"  # or "all"

# ---- INPUTS (change paths if you like) ----
GCMT_FILE = "./global_2/merged_gcmt_mechanisms.csv"
USGS_FILE = "./global_2/anss_mechanisms.csv"
ISC_FILE  = "./global_2/isc_mechanisms.txt"     # CSV with the same columns
GMT_FILE  = "./global_2/GMT_1976_2025_consolidado_conSlab.csv"
# ---- OUTPUT ----
OUT_MERGED = "merged_all_catalogs.csv"

# ---- Expected schema (what we keep) ----
FIELDS = [
    "id","time_iso","longitude","latitude","depth","mag","mag_type",
    "strike1","dip1","rake1","strike2","dip2","rake2",
    "T_plunge","T_azimuth","N_plunge","N_azimuth","P_plunge","P_azimuth",
    "Mrr","Mtt","Mpp","Mrt","Mrp","Mtp","source"
]

# ---- Source normalization & priority ----
def normalize_source(row: Dict[str, Any]) -> str:
    sid = str(row.get("id", "")).strip()
    src = (row.get("source") or "").strip().upper()

    # GCMT: explicit or classic C... id
    if "GCMT" in src or (sid.startswith("C") and any(ch.isalpha() for ch in sid[-1:])):
        return "GCMT"
    # USGS/ANSS
    if src in {"US","USGS","ANSS"} or sid.lower().startswith("us"):
        return "USGS"
    # GFZ/NEIC/NEIS
    if "GFZ" in src:  return "GFZ"
    if "NEIC" in src: return "NEIC"
    if "NEIS" in src: return "NEIS"
    # GMT (your 4th catalog) maps to OTHER unless you change priority below
    if "GMT" in src:  return "OTHER"
    return "OTHER"

PRIORITY = {
    "GCMT": 0,
    "USGS": 1,
    "GFZ":  2,
    "NEIC": 3,
    "NEIS": 4,
    "OTHER":5,
}


# ---- Helpers ----
def _sf(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)): return None
        return float(x)
    except Exception:
        return None

def haversine_km(lon1, lat1, lon2, lat2) -> float:
    if None in (lon1, lat1, lon2, lat2):
        return float("inf")
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl   = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def _prefer_record(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    ra, rb = normalize_source(a), normalize_source(b)
    pa, pb = PRIORITY.get(ra, 9), PRIORITY.get(rb, 9)
    if pa != pb:
        return a if pa < pb else b

    a_has_planes = all(_sf(a.get(k)) is not None for k in ("strike1","dip1","rake1","strike2","dip2","rake2"))
    b_has_planes = all(_sf(b.get(k)) is not None for k in ("strike1","dip1","rake1","strike2","dip2","rake2"))
    if a_has_planes != b_has_planes:
        return a if a_has_planes else b

    a_has_ten = any(_sf(a.get(k)) is not None for k in ("Mrr","Mtt","Mpp","Mrt","Mrp","Mtp"))
    b_has_ten = any(_sf(b.get(k)) is not None for k in ("Mrr","Mtt","Mpp","Mrt","Mrp","Mtp"))
    if a_has_ten != b_has_ten:
        return a if a_has_ten else b

    ma, mb = _sf(a.get("mag")), _sf(b.get("mag"))
    if (ma is not None) and (mb is not None) and (ma != mb):
        return a if ma > mb else b

    ta = pd.to_datetime(a.get("time_iso"), utc=True, errors="coerce")
    tb = pd.to_datetime(b.get("time_iso"), utc=True, errors="coerce")
    if pd.isna(ta) or pd.isna(tb):
        return a
    return a if ta <= tb else b

def _within_windows_time(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    ta = pd.to_datetime(a.get("time_iso"), utc=True, errors="coerce")
    tb = pd.to_datetime(b.get("time_iso"), utc=True, errors="coerce")
    if pd.isna(ta) or pd.isna(tb):
        return False
    return abs((tb - ta).total_seconds()) <= TIME_WINDOW_S

def _within_windows_space_mag(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    da = haversine_km(_sf(a.get("longitude")), _sf(a.get("latitude")),
                      _sf(b.get("longitude")), _sf(b.get("latitude")))
    if da > DIST_WINDOW_KM:
        return False
    ma, mb = _sf(a.get("mag")), _sf(b.get("mag"))
    if (ma is not None) and (mb is not None) and (abs(ma - mb) > MAG_WINDOW):
        return False
    return True

def _within_windows_all(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    # full criterion used among timed records
    return _within_windows_time(a, b) and _within_windows_space_mag(a, b)

# ---- Catalog loaders ----
def load_catalog_generic(path: str, source_hint: str = "") -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in FIELDS:
        if col not in df.columns:
            df[col] = np.nan
    if source_hint:
        df["source"] = df["source"].fillna(source_hint).replace("", source_hint)
    df["time_iso"] = df["time_iso"].astype(str)
    return df[FIELDS].copy()


def _time_from_gmt_id(eid: str) -> Optional[str]:
    """
    Parse time from GMT ID like '200503302331A' -> '2005-03-30T23:31:00'.
    Only applies to modern IDs starting with '20' and having 12 digits (+ optional trailing letter).
    Returns ISO string (UTC, seconds=00) or None if not parseable.
    """
    if not eid:
        return None
    s = str(eid).strip()
    # remove trailing letter if present
    if s and s[-1].isalpha():
        s = s[:-1]
    # must be 12 digits and start with '20' (YYYYMMDDhhmm)
    if not re.fullmatch(r"20\d{10}", s):
        return None
    try:
        dt = datetime.strptime(s, "%Y%m%d%H%M")
        return dt.isoformat(timespec="seconds")  # seconds = 00
    except Exception:
        return None

def load_gmt(path: str) -> pd.DataFrame:
    """
    Map GMT csv:
    ID,Lat,Lon,Depth_km,Mw,Strike_1,Dip_1,Rake_1,P_trend,P_plunge,T_trend,T_plunge,
    Strike_2,Dip_2,Rake_2,Strike_2_c,Dip_2_cal,Rake_2_cal,Slab1,Depth2Slab
    -> our FIELDS. time is recovered from ID when possible (post-2005 form).
    """
    raw = pd.read_csv(path)

    def pick(a, b):
        return a if pd.notna(a) else b

    # Prefer explicit second plane; fall back to *_cal if missing
    s2  = raw.get("Strike_2",   pd.Series([np.nan]*len(raw)))
    d2  = raw.get("Dip_2",      pd.Series([np.nan]*len(raw)))
    r2  = raw.get("Rake_2",     pd.Series([np.nan]*len(raw)))
    s2c = raw.get("Strike_2_c", pd.Series([np.nan]*len(raw)))
    d2c = raw.get("Dip_2_cal",  pd.Series([np.nan]*len(raw)))
    r2c = raw.get("Rake_2_cal", pd.Series([np.nan]*len(raw)))

    # recover times from ID when possible
    time_iso = [ _time_from_gmt_id(str(eid)) for eid in raw["ID"].astype(str) ]

    out = pd.DataFrame({
        "id":        raw["ID"].astype(str),
        "time_iso":  pd.Series(time_iso, dtype="object"),
        "longitude": raw["Lon"],
        "latitude":  raw["Lat"],
        "depth":     raw["Depth_km"],
        "mag":       raw["Mw"],
        "mag_type":  pd.Series(["Mw"]*len(raw)),
        "strike1":   raw["Strike_1"],
        "dip1":      raw["Dip_1"],
        "rake1":     raw["Rake_1"],
        "strike2":   [pick(s2.iloc[i], s2c.iloc[i]) for i in range(len(raw))],
        "dip2":      [pick(d2.iloc[i], d2c.iloc[i]) for i in range(len(raw))],
        "rake2":     [pick(r2.iloc[i], r2c.iloc[i]) for i in range(len(raw))],
        "T_plunge":  raw.get("T_plunge", pd.Series([np.nan]*len(raw))),
        "T_azimuth": raw.get("T_trend",  pd.Series([np.nan]*len(raw))),
        "N_plunge":  pd.Series([np.nan]*len(raw)),
        "N_azimuth": pd.Series([np.nan]*len(raw)),
        "P_plunge":  raw.get("P_plunge", pd.Series([np.nan]*len(raw))),
        "P_azimuth": raw.get("P_trend",  pd.Series([np.nan]*len(raw))),
        "Mrr":       pd.Series([np.nan]*len(raw)),
        "Mtt":       pd.Series([np.nan]*len(raw)),
        "Mpp":       pd.Series([np.nan]*len(raw)),
        "Mrt":       pd.Series([np.nan]*len(raw)),
        "Mrp":       pd.Series([np.nan]*len(raw)),
        "Mtp":       pd.Series([np.nan]*len(raw)),
        "source":    pd.Series(["GMT"]*len(raw)),
    })

    # numeric casts
    for col in ["longitude","latitude","depth","mag",
                "strike1","dip1","rake1","strike2","dip2","rake2",
                "T_plunge","T_azimuth","N_plunge","N_azimuth","P_plunge","P_azimuth",
                "Mrr","Mtt","Mpp","Mrt","Mrp","Mtp"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out[FIELDS].copy()
# ---- Clustering ----
def cluster_records(records: List[Dict[str, Any]]) -> List[List[int]]:
    """
    Cluster duplicates:
    - Timed records clustered with time+space+mag windows.
    - Untimed records are attached to any cluster if they match space+mag with ANY member (time ignored).
      Otherwise, each untimed record becomes its own cluster.
    """
    N = len(records)
    times = pd.to_datetime([r.get("time_iso") for r in records], utc=True, errors="coerce")
    ns = times.astype("int64")  # NaT -> -9223372036854775808
    order = np.argsort(ns, kind="mergesort")
    seconds = (ns.astype("float64") / 1e9).values
    seconds_sorted = seconds[order]

    timed_mask = ~pd.isna(times)
    timed_idx  = [i for i in order if timed_mask[i]]
    untimed_idx= [i for i in order if not timed_mask[i]]

    visited = np.zeros(N, dtype=bool)
    clusters: List[List[int]] = []

    # 1) Cluster timed records
    idx_to_pos = {order[pos]: pos for pos in range(N)}  # position in sorted order
    for idx in timed_idx:
        if visited[idx]:
            continue
        pos = idx_to_pos[idx]
        t0 = seconds_sorted[pos]

        # time window bounds in the sorted array
        left = pos
        while left - 1 >= 0 and seconds_sorted[left - 1] >= t0 - TIME_WINDOW_S:
            left -= 1
        right = pos
        while right + 1 < N and seconds_sorted[right + 1] <= t0 + TIME_WINDOW_S:
            right += 1

        cand = [order[k] for k in range(left, right + 1) if (order[k] in timed_idx and not visited[order[k]])]
        cluster = [idx]
        visited[idx] = True

        if DUP_CRITERION.lower() == "all":
            for j in cand:
                if j == idx or visited[j]: continue
                if _within_windows_all(records[idx], records[j]):
                    visited[j] = True
                    cluster.append(j)
        else:
            # 'any' union BFS inside time window
            frontier = [idx]
            available = set(cand); available.discard(idx)
            while frontier:
                k = frontier.pop()
                hits = [j for j in list(available) if _within_windows_all(records[k], records[j])]
                for j in hits:
                    available.remove(j)
                    visited[j] = True
                    cluster.append(j)
                    frontier.append(j)

        clusters.append(cluster)

    # 2) Attach untimed records by space+mag to ANY member of any existing cluster
    for idx in untimed_idx:
        if visited[idx]:
            continue
        attached = False
        for cl in clusters:
            # quick prune: compare to best (preferred) or 1st member; but safer to check any member
            for m in cl:
                if _within_windows_space_mag(records[idx], records[m]):
                    cl.append(idx)
                    visited[idx] = True
                    attached = True
                    break
            if attached:
                break
        if not attached:
            # its own cluster
            clusters.append([idx])
            visited[idx] = True

    return clusters

# ---- Merge driver ----
def merge_four(gcmt_path: str, usgs_path: str, isc_path: str, gmt_path: str, out_path: str):
    df_gcmt = load_catalog_generic(gcmt_path, source_hint="GCMT")
    df_usgs = load_catalog_generic(usgs_path)  # already has 'source'
    df_isc  = load_catalog_generic(isc_path)   # already has 'source'
    df_gmt  = load_gmt(gmt_path)

    all_df = pd.concat([df_gcmt, df_usgs, df_isc, df_gmt], ignore_index=True)
    records = all_df.to_dict("records")

    clusters = cluster_records(records)

    merged_rows: List[Dict[str, Any]] = []
    for cl in clusters:
        # pick preferred record
        best = records[cl[0]]
        for idx in cl[1:]:
            best = _prefer_record(best, records[idx])

        # collect duplicates
        dup_pairs = []
        best_id = str(best.get("id", ""))
        for idx in cl:
            rid = str(records[idx].get("id", ""))
            if rid == best_id:
                continue
            src = normalize_source(records[idx])
            dup_pairs.append(f"{src}:{rid}")

        out = {k: best.get(k, None) for k in FIELDS}
        out["source"] = normalize_source(best)
        out["dups"]   = ";".join(dup_pairs) if dup_pairs else ""
        merged_rows.append(out)

    out_df = pd.DataFrame(merged_rows)
    if not out_df.empty:
        out_df = out_df.sort_values(
            by="time_iso",
            key=lambda s: pd.to_datetime(s, utc=True, errors="coerce"),
            na_position="last"
        )
        cols = FIELDS.copy()
        if "dups" not in cols:
            cols.append("dups")
        out_df.to_csv(out_path, index=False, columns=cols)
    print(f"Wrote {out_path} with {len(out_df)} merged events "
          f"(from {len(all_df)} inputs across {len(clusters)} clusters).")

# ---- main ----
if __name__ == "__main__":
    merge_four(GCMT_FILE, USGS_FILE, ISC_FILE, GMT_FILE, OUT_MERGED)

# ---- main ---