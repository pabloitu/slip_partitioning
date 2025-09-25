import os
import re
from math import radians, sin, cos, asin, sqrt
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

FIELDS: List[str] = [
    "id", "time_iso", "longitude", "latitude", "depth", "mag", "mag_type",
    "lon_error", "lat_error", "depth_error", "mag_error",
    "strike1", "dip1", "rake1", "strike2", "dip2", "rake2",
    "Mrr", "Mtt", "Mpp", "Mrt", "Mrp", "Mtp",
    "source", "dups",
]

MOMENT_FIELDS = ("Mrr", "Mtt", "Mpp", "Mrt", "Mrp", "Mtp")
SDR_FIELDS    = ("strike1", "dip1", "rake1", "strike2", "dip2", "rake2")

CANON_SOURCE: Dict[str, str] = {
    "gcmt": "gcmt", "cmt": "gcmt",
    "anss": "anss", "us": "anss", "usgs": "anss",
}

PreferenceRule = Tuple[int, Optional[float], Optional[float]]
PreferenceSpec = Dict[str, Union[int, Iterable[PreferenceRule]]]

def _finite(x: Any) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def _pair_km_threshold(src_i: str, src_j: str, mag_i: Any, mag_j: Any, base_km: float) -> float:
    """Return a magnitude/source-aware km threshold for clustering."""
    try:
        m_i = float(mag_i)
    except Exception:
        m_i = np.nan
    try:
        m_j = float(mag_j)
    except Exception:
        m_j = np.nan

    maxm = np.nanmax([m_i, m_j])
    thr = float(base_km)

    if np.isfinite(maxm):
        if maxm >= 7.8:
            thr = max(thr, 220.0)
        elif maxm >= 7.2:
            thr = max(thr, 150.0)
        elif maxm >= 6.8:
            thr = max(thr, 100.0)

    sset = {str(src_i).lower(), str(src_j).lower()}
    if sset == {"gcmt", "anss"}:
        thr = max(thr, base_km + 40.0)  # small universal bump

    return thr

def ensure_fields(df: pd.DataFrame) -> pd.DataFrame:
    needed = set(FIELDS) - {"dups"}
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan
    if "source" not in df.columns:
        df["source"] = np.nan
    return df

def load_catalog_forced_source(path: str, forced_source_key: str) -> pd.DataFrame:
    """Load CSV, enforce unified schema, and **force** source to the given key."""
    df = pd.read_csv(path)
    df = ensure_fields(df)
    df["source"] = forced_source_key.strip().lower()
    df["source"] = df["source"].map(lambda s: CANON_SOURCE.get(s, s))
    df["time_iso"] = df["time_iso"].astype(str)
    return df

def canonical_id(s: Any) -> str:
    _LETTER_DIGIT = re.compile(r'^[A-Za-z](\d.*)$')
    s = (str(s or "")).strip().lower()
    m = _LETTER_DIGIT.match(s)
    return m.group(1) if m else s

def haversine_km(lon1: Any, lat1: Any, lon2: Any, lat2: Any) -> float:
    if not all(_finite(v) for v in (lon1, lat1, lon2, lat2)):
        return np.inf
    lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2
    return 2.0 * 6371.0 * asin(sqrt(a))

def normalize_source(row: Dict[str, Any]) -> str:
    src = (row.get("source") or "").strip().lower()
    return CANON_SOURCE.get(src, src) if src else "other"

def _to_unix_seconds(series: pd.Series) -> np.ndarray:
    t = pd.to_datetime(series, utc=True, errors="coerce")
    arr = np.full(len(series), np.nan, dtype="float64")
    mask = t.notna()
    if mask.any():
        try:
            arr[mask.to_numpy()] = (t[mask].view("int64") // 10**9).astype(float)
        except Exception:
            arr[mask.to_numpy()] = (t[mask].astype("int64") // 10**9).astype(float)
    return arr

def _record_epoch(rec: Dict[str, Any]) -> Optional[float]:
    try:
        ts = pd.to_datetime(rec.get("time_iso"), utc=True, errors="coerce")
        if pd.notna(ts):
            return float(int(ts.value // 10**9))
        return None
    except Exception:
        return None

# -------- preference ranking over EPOCH --------
def _rank_for_source_epoch(source: str,
                           epoch: Optional[float],
                           preference: PreferenceSpec,
                           other_default: int = 999) -> Tuple[int, int]:
    """
    Return (rank, tie_order) for a given source and event epoch using 'preference'.
    - Simple: {"anss": 0, "gcmt": 1}
    - Ranged: {"anss": ((0, 1136073600, None), (1, None, 1136073599)), ...}
      epochs are in seconds since 1970-01-01 UTC; ranges inclusive; None=open.
    If epoch is None with ranged rules, the best (lowest) rank among rules is used.
    """
    src = (source or "other").lower()
    spec = preference.get(src)
    if spec is None:
        return (other_default, 10**6)

    if isinstance(spec, (int, np.integer)):
        return (int(spec), 0)

    best_rank = other_default
    best_tie = 10**6
    for idx, rule in enumerate(spec):  # type: ignore[union-attr]
        pr, e0, e1 = rule
        e0 = float(e0) if e0 is not None else None
        e1 = float(e1) if e1 is not None else None
        if epoch is None:
            if pr < best_rank or (pr == best_rank and idx < best_tie):
                best_rank, best_tie = int(pr), idx
            continue
        if (e0 is None or epoch >= e0) and (e1 is None or epoch <= e1):
            return (int(pr), idx)
    return (best_rank, best_tie)

def _prefer(a_rank: Tuple[int,int], b_rank: Tuple[int,int]) -> bool:
    if a_rank[0] != b_rank[0]:
        return a_rank[0] < b_rank[0]
    return a_rank[1] < b_rank[1]

# -------------------- clustering --------------------
class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1
# --- REPLACE your dedup_clusters with this version ---
def dedup_clusters(records: List[Dict[str, Any]],
                   dt_near_s: float,
                   km_near: float,
                   dmag_near: float) -> List[List[int]]:
    n = len(records)
    dsu = DSU(n)

    # R1: same canonical id
    by_id: Dict[str, List[int]] = {}
    for i, r in enumerate(records):
        cid = canonical_id(r.get("id"))
        if cid:
            by_id.setdefault(cid, []).append(i)
    for idxs in by_id.values():
        for a, b in zip(idxs, idxs[1:]):
            dsu.union(a, b)

    # R2: near in time/space/mag (with dynamic space threshold)
    series_time = pd.Series([r.get("time_iso") for r in records], dtype="object")
    times_sec = _to_unix_seconds(series_time)
    lons = np.array([r.get("longitude") for r in records], dtype=object)
    lats = np.array([r.get("latitude")  for r in records], dtype=object)
    mags = np.array([r.get("mag")       for r in records], dtype=object)
    srcs = np.array([normalize_source(r) for r in records], dtype=object)

    valid_t = np.isfinite(times_sec)
    order = np.argsort(np.where(valid_t, times_sec, np.inf))

    for pos, i in enumerate(order):
        if not valid_t[i]:
            continue
        ti = times_sec[i]
        jpos = pos + 1
        while jpos < n:
            j = order[jpos]
            if not valid_t[j]:
                jpos += 1
                continue
            dt = times_sec[j] - ti
            if dt > dt_near_s:
                break

            # All needed fields numeric?
            if all(_finite(x) for x in (lons[i], lats[i], lons[j], lats[j], mags[i], mags[j])):
                dkm = haversine_km(lons[i], lats[i], lons[j], lats[j])
                dmag = abs(float(mags[i]) - float(mags[j]))

                # magnitude/source-aware spatial threshold
                km_thr = _pair_km_threshold(srcs[i], srcs[j], mags[i], mags[j], km_near)

                if (dkm <= km_thr) and (dmag <= dmag_near):
                    dsu.union(i, j)
            jpos += 1

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        root = dsu.find(i)
        groups.setdefault(root, []).append(i)
    return list(groups.values())


# -------------------- field selection --------------------
def _has_required_fields(rec: Dict[str, Any], fields: Sequence[str]) -> bool:
    for k in fields:
        v = rec.get(k, None)
        if k in MOMENT_FIELDS:
            if not _finite(v):
                return False
        else:
            if v is None:
                return False
            if isinstance(v, str):
                if v.strip() == "":
                    return False
            else:
                if not _finite(v):
                    return False
    return True

def _best_record_by_preference(records: List[Dict[str, Any]],
                               idxs: List[int],
                               preference: PreferenceSpec) -> Dict[str, Any]:
    """Pick the single best record by epoch-based preference (no field requirements)."""
    best = None
    best_key = (10**9, 10**6)
    for i in idxs:
        rec = records[i]
        src   = normalize_source(rec)
        epoch = _record_epoch(rec)
        key   = _rank_for_source_epoch(src, epoch, preference)
        if _prefer(key, best_key):
            best = rec
            best_key = key
    return best if best is not None else records[idxs[0]]

def _pick_with_preferred_source(records: List[Dict[str, Any]],
                                idxs: List[int],
                                required_fields: Sequence[str],
                                preference: PreferenceSpec,
                                preferred_source: str) -> Dict[str, Any]:
    best = None; best_key = (10**9, 10**6)
    for i in idxs:
        rec = records[i]
        if normalize_source(rec) != preferred_source:
            continue
        if not _has_required_fields(rec, required_fields):
            continue
        key = _rank_for_source_epoch(normalize_source(rec), _record_epoch(rec), preference)
        if _prefer(key, best_key):
            best = rec; best_key = key
    if best is not None:
        return best

    best = None; best_key = (10**9, 10**6)
    for i in idxs:
        rec = records[i]
        if not _has_required_fields(rec, required_fields):
            continue
        key = _rank_for_source_epoch(normalize_source(rec), _record_epoch(rec), preference)
        if _prefer(key, best_key):
            best = rec; best_key = key
    if best is not None:
        return best

    best_pref_src = None; best_key = (10**9, 10**6)
    for i in idxs:
        rec = records[i]
        if normalize_source(rec) == preferred_source:
            key = _rank_for_source_epoch(normalize_source(rec), _record_epoch(rec), preference)
            if _prefer(key, best_key):
                best_pref_src = rec; best_key = key
    if best_pref_src is not None:
        return best_pref_src

    return _best_record_by_preference(records, idxs, preference)

def assemble_output_record(records: List[Dict[str, Any]],
                           idxs: List[int],
                           preference: PreferenceSpec) -> Dict[str, Any]:
    base_rec = _best_record_by_preference(records, idxs, preference)
    preferred_source = normalize_source(base_rec)

    loc_rec = _pick_with_preferred_source(
        records, idxs,
        required_fields=("longitude","latitude","depth"),
        preference=preference,
        preferred_source=preferred_source,
    )

    mag_rec = _pick_with_preferred_source(
        records, idxs,
        required_fields=("mag","mag_type"),
        preference=preference,
        preferred_source=preferred_source,
    )

    tens_rec = _pick_with_preferred_source(
        records, idxs, required_fields=MOMENT_FIELDS,
        preference=preference, preferred_source=preferred_source
    )
    sdr_rec = _pick_with_preferred_source(
        records, idxs, required_fields=SDR_FIELDS,
        preference=preference, preferred_source=preferred_source
    )

    out: Dict[str, Any] = {
        "id":        base_rec.get("id"),
        "time_iso":  base_rec.get("time_iso"),
        "source":    preferred_source,
        "longitude": loc_rec.get("longitude"),
        "latitude":  loc_rec.get("latitude"),
        "depth":     loc_rec.get("depth"),
        "mag":       mag_rec.get("mag"),
        "mag_type":  (mag_rec.get("mag_type").lower()
                      if isinstance(mag_rec.get("mag_type"), str) else mag_rec.get("mag_type")),
        "lon_error":   loc_rec.get("lon_error"),
        "lat_error":   loc_rec.get("lat_error"),
        "depth_error": loc_rec.get("depth_error"),
        "mag_error":   mag_rec.get("mag_error"),
    }
    for k in MOMENT_FIELDS:
        out[k] = tens_rec.get(k) if tens_rec is not None else None
    for k in SDR_FIELDS:
        out[k] = sdr_rec.get(k) if sdr_rec is not None else None

    chosen_id = str(out.get("id", ""))
    dups = []
    for i in idxs:
        rid = str(records[i].get("id", ""))
        if rid and rid != chosen_id:
            dups.append(f"{normalize_source(records[i])}:{rid}")
    out["dups"] = ";".join(dups)
    return out

def assign_duplicate_ids(records: List[Dict[str, Any]],
                         clusters: List[List[int]]) -> List[str]:
    def rec_time(idx: int) -> int:
        t = pd.to_datetime(records[idx].get("time_iso"), utc=True, errors="coerce")
        return int(t.value) if pd.notna(t) else -2**62

    clusters_sorted = sorted(clusters, key=lambda cl: min(rec_time(i) for i in cl))
    dup_id_by_idx = [""] * len(records)
    for cluster_num, cl in enumerate(clusters_sorted, start=1):
        per_src: Dict[str, int] = {}
        for i in cl:
            src = normalize_source(records[i]) or "other"
            count = per_src.get(src, 0)
            suffix = src if count == 0 else f"{src}{chr(ord('a') + count)}"
            dup_id_by_idx[i] = f"{cluster_num}_{suffix}"
            per_src[src] = count + 1
    return dup_id_by_idx

def merge_and_label(input_files: Dict[str, str],
                    out_merged_path: str,
                    out_full_path: str,
                    dt_near_s: float = 120.0,
                    km_near: float = 80.0,
                    dmag_near: float = 0.8,
                    preference: PreferenceSpec | None = None) -> None:
    """
    examples:
      # constant preference (prefer ANSS)
      {"anss": 0, "gcmt": 1}

      # ranged preference (inclusive), epochs in seconds
      {
        "anss": ((0, 1136073600, None), (1, None, 1136073599)),
        "gcmt": ((1, 1136073600, None), (0, None, 1136073599)),
      }
    """
    if preference is None:
        preference = {"anss": 0, "gcmt": 1}

    frames: List[pd.DataFrame] = []
    for key, path in input_files.items():
        key_norm = CANON_SOURCE.get(key.lower(), key.lower())
        frames.append(load_catalog_forced_source(path, forced_source_key=key_norm))

    all_df = pd.concat(frames, ignore_index=True, sort=False)
    all_df["source"] = all_df["source"].astype(str).str.strip().str.lower().map(lambda s: CANON_SOURCE.get(s, s))

    records = all_df.to_dict("records")
    clusters = dedup_clusters(records, dt_near_s=dt_near_s, km_near=km_near, dmag_near=dmag_near)

    merged_rows = [assemble_output_record(records, cl, preference=preference) for cl in clusters]
    df_merged = pd.DataFrame(merged_rows)

    if not df_merged.empty:
        df_merged = df_merged.sort_values(
            by="time_iso",
            key=lambda s: pd.to_datetime(s, utc=True, errors="coerce"),
            na_position="last",
        )
        cols = [c for c in FIELDS if c in df_merged.columns]
        os.makedirs(os.path.dirname(out_merged_path) or ".", exist_ok=True
        )
        df_merged.to_csv(out_merged_path, index=False, columns=cols)

    dup_ids = assign_duplicate_ids(records, clusters)
    df_full = all_df.copy()
    df_full["source"] = df_full["source"].map(lambda s: CANON_SOURCE.get(str(s).lower(), str(s).lower()))
    df_full["duplicate_id"] = dup_ids
    df_full = df_full.sort_values(
        by="time_iso",
        key=lambda s: pd.to_datetime(s, utc=True, errors="coerce"),
        na_position="last",
    )
    os.makedirs(os.path.dirname(out_full_path) or ".", exist_ok=True)
    df_full.to_csv(out_full_path, index=False)

    print(f"merged -> {out_merged_path}  ({len(df_merged)} rows)")
    print(f"full+dups -> {out_full_path}  ({len(df_full)} rows)")

# -------------------- Main (example) --------------------
def _epoch_of_ymd(year: int, month: int, day: int) -> int:
    return int(pd.Timestamp(year=year, month=month, day=day, tz="UTC").value // 10**9)

def main() -> None:
    from cat_handler import paths

    input_files = {
        "gcmt": str(paths.cat_gcmt),   # GCMT CSV parsed to unified schema
        "anss": str(paths.cat_anss),   # ANSS CSV parsed to unified schema
    }

    # Example A: simple constant preference (prefer ANSS everywhere)
    preference = {"anss": 0, "gcmt": 1}

    # cutoff = _epoch_of_ymd(2006, 1, 1)
    # preference = {
        # "anss": ((0, cutoff, None), (1, None, cutoff - 1)),
        # "gcmt": ((1, cutoff, None), (0, None, cutoff - 1)),
    # }

    merge_and_label(
        input_files=input_files,
        out_merged_path=str(paths.cat_merged),
        out_full_path=str(paths.cat_full),
        dt_near_s=120.0,
        km_near=80.0,
        dmag_near=0.8,
        preference=preference,
    )

if __name__ == "__main__":
    main()
