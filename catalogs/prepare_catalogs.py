#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare per-source catalogs into a common schema and filter them consistently.

Global filters applied to *all* outputs:
  - LAT_MIN <= latitude <= LAT_MAX
  - LON_MIN <= longitude <= LON_MAX
  - time_iso >= TIME_MIN  (1976-01-01)
  - mag >= MAG_MIN  (4.95)

NOTE on missing times:
  STRICT_TIME_FILTER controls whether rows with missing/invalid time are dropped.
  - False (default): keep rows with NaT (cannot test time); only enforce cutoff where time is present.
  - True: drop rows whose time cannot be parsed to >= TIME_MIN.

Outputs go to the 'processed_catalogs' folder.
"""

from __future__ import annotations

import os
import io
import re
import math
import gzip
import time
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from obspy import read_events
from libcomcat.search import get_event_by_id

# ---------------- GLOBAL FILTERS (EDIT HERE) ----------------
LAT_MIN, LAT_MAX = -54.0, -17.0
LON_MIN, LON_MAX = -80.0, -65.0
TIME_MIN_STR = "1976-01-01T00:00:00"
MAG_MIN = 4.95
STRICT_TIME_FILTER = False  # True => drop rows without a valid time

TIME_MIN = pd.to_datetime(TIME_MIN_STR, utc=True)

# ---------------- COMMON OUTPUT SCHEMA ----------------
FIELDS = [
    "id","time_iso","longitude","latitude","depth","mag","mag_type",
    "strike1","dip1","rake1","strike2","dip2","rake2",
    "T_plunge","T_azimuth","N_plunge","N_azimuth","P_plunge","P_azimuth",
    "Mrr","Mtt","Mpp","Mrt","Mrp","Mtp","source",
]

# ---------------- SMALL UTILS ----------------
def _sf(x) -> Optional[float]:
    try:
        f = float(x)
        return f if np.isfinite(f) else None
    except Exception:
        return None

def _finite(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def ensure_fields_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in FIELDS:
        if c not in df.columns:
            df[c] = np.nan
    return df

def filter_df(df: pd.DataFrame,
              strict_time: bool = STRICT_TIME_FILTER) -> pd.DataFrame:
    """
    Apply spatial, temporal, and magnitude filters uniformly.
    - Spatial: requires finite lon/lat within bounds.
    - Magnitude: requires finite mag >= MAG_MIN.
    - Time: if strict_time=True, require parsed time >= TIME_MIN.
            if strict_time=False, keep rows with missing time, but enforce cutoff where time is present.
    """
    df = df.copy()

    # Numeric coercion for spatial + mag
    for col in ["longitude", "latitude", "mag"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Spatial mask
    m_spatial = (
        df["latitude"].between(LAT_MIN, LAT_MAX) &
        df["longitude"].between(LON_MIN, LON_MAX)
    )

    # Magnitude mask (require finite mag and >= MAG_MIN)
    m_mag = pd.to_numeric(df["mag"], errors="coerce").ge(MAG_MIN)

    # Time mask
    if "time_iso" in df.columns:
        t = pd.to_datetime(df["time_iso"], utc=True, errors="coerce")
        if strict_time:
            m_time = t.ge(TIME_MIN)
        else:
            # keep NaT; enforce only where time is present
            m_time = t.ge(TIME_MIN) | t.isna()
    else:
        # no time column -> keep all if not strict; drop all if strict
        m_time = (~strict_time)

    mask = m_spatial & m_mag & m_time
    out = df.loc[mask].copy()

    # sort by time if present
    if "time_iso" in out.columns:
        out = out.sort_values(
            by="time_iso",
            key=lambda s: pd.to_datetime(s, utc=True, errors="coerce"),
            na_position="last"
        )
    return out

# =========================================================
# -----------------------  ISC  ---------------------------
# =========================================================
def _clean_token(s: str) -> str:
    s = s.strip()
    if "<" in s:
        s = s.split("<", 1)[0].strip()
    return s

def _to_iso(date_s: str, time_s: str) -> Optional[str]:
    s = f"{date_s.strip()} {time_s.strip()}"
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).isoformat(timespec="seconds")
        except ValueError:
            continue
    return None

def _to_nm(val: float | None, exponent: int | None) -> float | None:
    if val is None or exponent is None:
        return None
    return val * (10.0 ** exponent) * 1e-7  # dyne·cm -> N·m

def parse_isc_file(path: str, PREF_RANK: dict) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]

    header_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("EVENT_ID,"):
            header_idx = i
            break
    if header_idx is None:
        print("Header row not found in ISC file.")
        return rows

    header = [h.strip() for h in lines[header_idx].split(",")]
    idx = {name: header.index(name) for name in header}

    def get(rowlist, name, default=""):
        try:
            return rowlist[idx[name]]
        except Exception:
            return default

    for ln in lines[header_idx + 1:]:
        if not ln.strip():
            continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < len(header):
            continue

        raw_event = get(parts, "EVENT_ID")
        if not raw_event:
            continue
        event_id = _clean_token(raw_event).split()[0]

        date_s = get(parts, "DATE")
        time_s = get(parts, "TIME")
        time_iso = _to_iso(date_s, time_s)

        lat = _sf(get(parts, "LAT"))
        lon = _sf(get(parts, "LON"))
        depth = _sf(get(parts, "DEPTH"))

        author_positions = [i for i, h in enumerate(header) if h == "AUTHOR"]
        fm_author = _clean_token(parts[author_positions[1]]) if len(author_positions) >= 2 else _clean_token(get(parts, "AUTHOR", ""))
        fm_author_up = fm_author.upper()
        if fm_author_up == "GCMT":
            continue

        mw_str = get(parts, "MW", "").strip()
        mag = _sf(mw_str)
        mag_type = "mw" if mag is not None else None

        ex_comp = None
        try:
            ex_comp = int(get(parts, "EX"))
        except Exception:
            ex_positions = [i for i, h in enumerate(header) if h == "EX"]
            if len(ex_positions) >= 2:
                try:
                    ex_comp = int(parts[ex_positions[1]])
                except Exception:
                    ex_comp = None

        Mrr = _to_nm(_sf(get(parts, "MRR")), ex_comp)
        Mtt = _to_nm(_sf(get(parts, "MTT")), ex_comp)
        Mpp = _to_nm(_sf(get(parts, "MPP")), ex_comp)
        Mrt = _to_nm(_sf(get(parts, "MRT")), ex_comp)
        Mtp = _to_nm(_sf(get(parts, "MTP")), ex_comp)
        Mrp = _to_nm(_sf(get(parts, "MPR")), ex_comp)

        # nodal planes (two copies in header)
        strike_positions = [i for i, h in enumerate(header) if h == "STRIKE"]
        dip_positions    = [i for i, h in enumerate(header) if h == "DIP"]
        rake_positions   = [i for i, h in enumerate(header) if h == "RAKE"]

        s1 = _sf(get(parts, "STRIKE"))
        d1 = _sf(get(parts, "DIP"))
        r1 = _sf(get(parts, "RAKE"))

        if len(strike_positions) >= 2 and len(dip_positions) >= 2 and len(rake_positions) >= 2:
            s2 = _sf(parts[strike_positions[1]])
            d2 = _sf(parts[dip_positions[1]])
            r2 = _sf(parts[rake_positions[1]])
        else:
            s2 = d2 = r2 = None

        T_pl = _sf(get(parts, "T_PL"))
        T_az = _sf(get(parts, "T_AZM"))
        N_pl = _sf(get(parts, "N_PL"))
        N_az = _sf(get(parts, "N_AZM"))
        P_pl = _sf(get(parts, "P_PL"))
        P_az = _sf(get(parts, "P_AZM"))

        rows.append(dict(
            id=str(event_id),
            time_iso=time_iso,
            longitude=lon, latitude=lat, depth=depth,
            mag=mag, mag_type=mag_type,
            strike1=s1, dip1=d1, rake1=r1,
            strike2=s2, dip2=d2, rake2=r2,
            T_plunge=T_pl, T_azimuth=T_az,
            N_plunge=N_pl, N_azimuth=N_az,
            P_plunge=P_pl, P_azimuth=P_az,
            Mrr=Mrr, Mtt=Mtt, Mpp=Mpp, Mrt=Mrt, Mrp=Mrp, Mtp=Mtp,
            source=fm_author,
            _rank=PREF_RANK.get(fm_author_up, 9)
        ))
    return rows

def choose_best_per_event(all_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_rows: by_id.setdefault(r["id"], []).append(r)

    def filled_score(rec: Dict[str, Any]) -> int:
        keys = ["strike1","dip1","rake1","strike2","dip2","rake2",
                "T_plunge","T_azimuth","N_plunge","N_azimuth","P_plunge","P_azimuth",
                "Mrr","Mtt","Mpp","Mrt","Mrp","Mtp"]
        return sum(1 for k in keys if rec.get(k) is not None)

    out = []
    for eid, group in by_id.items():
        group.sort(key=lambda r: (r.get("_rank", 9), ), reverse=False)
        best_rank = group[0].get("_rank", 9)
        cands = [g for g in group if g.get("_rank", 9) == best_rank]
        cands.sort(key=lambda r: (filled_score(r), r.get("time_iso") or ""), reverse=True)
        chosen = cands[0].copy()
        chosen.pop("_rank", None)
        out.append(chosen)

    out.sort(key=lambda r: r.get("time_iso") or "")
    return out

def write_csv(rows: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            rec = {k: r.get(k, None) for k in FIELDS}
            w.writerow(rec)

def prepare_isc(in_path: str, out_path: str, priority: Dict[str,int]):
    rows = parse_isc_file(in_path, priority)
    if not rows:
        print("[ISC] no rows parsed")
        return
    selected = choose_best_per_event(rows)
    df = pd.DataFrame(selected, columns=FIELDS)
    df = filter_df(df)  # <-- unified filter
    df.to_csv(out_path, index=False)
    print(f"[ISC] wrote {out_path}  ({len(df)} rows)")

# =========================================================
# -----------------------  ANSS  --------------------------
# =========================================================
_last_request_ts = 0.0

def _pause(min_interval: float):
    global _last_request_ts
    now = time.monotonic()
    dt = now - _last_request_ts
    if dt < min_interval:
        time.sleep(min_interval - dt)
    _last_request_ts = time.monotonic()

def _is_quakeml_bytes(b: bytes) -> bool:
    if not b: return False
    head = b[:4096].lower()
    return (b"<?xml" in head or b"<q:quakeml" in head or b"<quakeml" in head)

def _maybe_decompress(b: bytes) -> bytes:
    if not b: return b
    if len(b) >= 2 and b[0] == 0x1F and b[1] == 0x8B:
        try: return gzip.decompress(b)
        except Exception: return b
    return b

def _to_bytes(obj):
    if isinstance(obj, tuple) and obj: obj = obj[0]
    if isinstance(obj, (bytes, bytearray)): return bytes(obj)
    if isinstance(obj, str): return obj.encode("utf-8", "ignore")
    if hasattr(obj, "read"):
        b = obj.read()
        return b if isinstance(b, (bytes, bytearray)) else str(b).encode("utf-8", "ignore")
    return None

def _axes_from_sdr(strike, dip, rake):
    st, di, ra = np.radians([strike, dip, rake])
    n = np.array([-np.sin(di)*np.sin(st),  np.sin(di)*np.cos(st), -np.cos(di)])
    s = np.array([ np.cos(ra)*np.cos(st)+np.sin(ra)*np.cos(di)*np.sin(st),
                   np.cos(ra)*np.sin(st)-np.sin(ra)*np.cos(di)*np.cos(st),
                   np.sin(ra)*np.sin(di)])
    n/=np.linalg.norm(n); s/=np.linalg.norm(s)
    M = np.outer(s,n)+np.outer(n,s)
    w,V = np.linalg.eigh(M); V = V[:, np.argsort(w)]  # P,N,T
    def azpl(v):
        v = v/np.linalg.norm(v)
        if v[2] < 0: v = -v
        az = (math.degrees(math.atan2(v[1], v[0])) + 360) % 360
        pl = math.degrees(math.asin(max(-1, min(1, v[2]))))
        return az, pl
    P,N,T = V[:,0],V[:,1],V[:,2]
    T_az,T_pl = azpl(T); N_az,N_pl = azpl(N); P_az,P_pl = azpl(P)
    return T_pl,T_az,N_pl,N_az,P_pl,P_az

def _sdr_from_tensor_rtp(mrr,mtt,mpp,mrt,mrp,mtp):
    Mrtp = np.array([[mrr,mrt,mrp],[mrt,mtt,mtp],[mrp,mtp,mpp]], float)
    A    = np.array([[0,-1,0],[0,0,1],[-1,0,0]], float)
    Mned = A @ Mrtp @ A.T
    w,V  = np.linalg.eigh(Mned); V = V[:, np.argsort(w)]
    P,N,T = V[:,0],V[:,1],V[:,2]
    def v(az,pl):
        az,pl=np.radians([az,pl]); return np.array([np.cos(pl)*np.cos(az),np.cos(pl)*np.sin(az),np.sin(pl)])
    def azpl(x):
        if x[2] < 0: x = -x
        az = (math.degrees(math.atan2(x[1], x[0])) + 360) % 360
        pl = math.degrees(math.asin(max(-1, min(1, x[2]))))
        return az, pl
    T_az,T_pl = azpl(T); P_az,P_pl = azpl(P)
    T_hat, P_hat = v(T_az,T_pl), v(P_az,P_pl)
    n1 = T_hat + P_hat; s1 = T_hat - P_hat
    n2 = T_hat - P_hat; s2 = T_hat + P_hat
    def n_s_to_sdr(n,s):
        n/=np.linalg.norm(n); s/=np.linalg.norm(s)
        if n[2] > 0: n=-n; s=-s
        dip = math.degrees(math.atan2(abs(n[2]), math.hypot(n[0], n[1])))
        strike = (math.degrees(math.atan2(n[1], n[0])) + 90) % 360
        phi = math.radians(strike)
        u_strike = np.array([-math.sin(phi), math.cos(phi), 0.0])
        delta = math.radians(dip)
        u_dip = np.array([math.cos(phi)*math.sin(delta), math.sin(phi)*math.sin(delta), -math.cos(delta)])
        rake = math.degrees(math.atan2(float(np.dot(s,u_dip)), float(np.dot(s,u_strike))))
        if rake > 180: rake -= 360
        if rake <= -180: rake += 360
        return strike,dip,rake
    return n_s_to_sdr(n1,s1), n_s_to_sdr(n2,s2)

def _preferred_origin(ev):
    try:
        for fm in (ev.focal_mechanisms or []):
            mt = getattr(fm, "moment_tensor", None)
            oid = str(getattr(mt, "derived_origin_id", "")) if mt else ""
            if not oid: continue
            for org in (ev.origins or []):
                if str(getattr(org, "resource_id", "")) == oid and getattr(org, "time", None):
                    return org
    except Exception:
        pass
    for org in (ev.origins or []):
        if getattr(org, "time", None):
            return org
    return ev.preferred_origin() or (ev.origins[0] if (ev.origins or []) else None)

def get_quakeml_bytes(row,
                      min_interval: float = 0.35,
                      tries_event: int = 2,
                      tries_content: int = 2,
                      backoff_base: float = 2.0) -> bytes | None:
    eid = str(row["id"])
    _pause(min_interval)
    ev = None
    for k in range(tries_event):
        try:
            ev = get_event_by_id(eventid=eid, includesuperseded=True)
            break
        except Exception as e:
            if k == tries_event - 1:
                print(f"[{eid}] get_event_by_id failed after {tries_event} tries: {e}")
                return None
            time.sleep((backoff_base ** k) + random.uniform(0, 0.5))
    if ev is None:
        return None

    def _safe_get(ev, name):
        try:
            return ev.getProducts(name) or []
        except Exception:
            return []

    def _collect(ptypes):
        out = []
        for ptype in ptypes:
            prods = _safe_get(ev, ptype)
            if not prods:
                continue
            us = [p for p in prods if getattr(p, "source", None) == "us"]
            out += (us if us else prods)
        return out

    candidates = _collect(("focal-mechanism", "moment-tensor")) or _collect(("phase-data","origin"))
    if not candidates:
        return None

    def _candidate_names(prod):
        names = []
        try:
            for key in getattr(prod, "contents", []):
                lk = key.lower()
                if ("quakeml" in lk and lk.endswith(".xml")) or lk.endswith(".xml.gz"):
                    names.append(key)
        except Exception:
            pass
        names = sorted(set(["quakeml.xml", "quakeml"] + names), key=len)
        return names

    for prod in candidates:
        names = _candidate_names(prod) or ["quakeml.xml","quakeml"]
        for k in range(tries_content):
            for name in names:
                try:
                    b = _to_bytes(prod.getContentBytes(name))
                    b = _maybe_decompress(b)
                    if b and _is_quakeml_bytes(b):
                        return b
                except Exception:
                    continue
            if k < tries_content - 1:
                time.sleep((backoff_base ** k) + random.uniform(0, 0.5))
    print(f"[{eid}] FM/MT present but no usable QuakeML content")
    return None

def parse_quakeml_row(xml_bytes: bytes, row) -> dict | None:
    cat = read_events(io.BytesIO(xml_bytes))
    if not cat:
        return None
    ev = cat[0]
    org = _preferred_origin(ev)

    time_iso = None
    if org and getattr(org, "time", None):
        try:
            time_iso = org.time.datetime.replace(tzinfo=None).isoformat(timespec="seconds")
        except Exception:
            time_iso = None
    if time_iso is None:
        try:
            t = pd.to_datetime(row.get("time") or row.get("time_iso"), utc=True, errors="coerce")
            if pd.notna(t):
                time_iso = t.tz_convert(None).isoformat(timespec="seconds")
        except Exception:
            time_iso = None

    lon = _sf(getattr(org, "longitude", None)) if org and getattr(org, "longitude", None) is not None else _sf(row.get("longitude"))
    lat = _sf(getattr(org, "latitude",  None)) if org and getattr(org, "latitude",  None) is not None else _sf(row.get("latitude"))
    if org and getattr(org, "depth", None) is not None:
        depth = _sf(org.depth) / 1000.0
    else:
        depth = _sf(row.get("depth"))

    pref_order = ["mww", "mwc", "mwr", "mwb", "mw"]
    mag = None; mag_type = None; best_rank = 10**9
    for m in (getattr(ev, "magnitudes", None) or []):
        t = (getattr(m, "magnitude_type", None) or "").strip().lower()
        if not t.startswith("mw") or m.mag is None:
            continue
        rank = pref_order.index(t) if t in pref_order else pref_order.index("mw")
        if rank < best_rank:
            best_rank = rank
            mag = float(m.mag); mag_type = t
    if mag is None:
        mag = _sf(row.get("mag"))
        mt = row.get("magType") or row.get("mag_type")
        mag_type = (str(mt).strip().lower() if mt is not None else None)

    s1 = d1 = r1 = s2 = d2 = r2 = None
    T_pl = T_az = N_pl = N_az = P_pl = P_az = None
    Mrr = Mtt = Mpp = Mrt = Mrp = Mtp = None

    fm = ev.focal_mechanisms[0] if ev.focal_mechanisms else None
    if fm:
        npn = getattr(fm, "nodal_planes", None)
        if npn and npn.nodal_plane_1 and npn.nodal_plane_2:
            s1,d1,r1 = float(npn.nodal_plane_1.strike), float(npn.nodal_plane_1.dip), float(npn.nodal_plane_1.rake)
            s2,d2,r2 = float(npn.nodal_plane_2.strike), float(npn.nodal_plane_2.dip), float(npn.nodal_plane_2.rake)
        pax = getattr(fm, "principal_axes", None)
        if pax and pax.t_axis and pax.n_axis and pax.p_axis:
            T_pl,T_az = _sf(pax.t_axis.plunge), _sf(pax.t_axis.azimuth)
            N_pl,N_az = _sf(pax.n_axis.plunge), _sf(pax.n_axis.azimuth)
            P_pl,P_az = _sf(pax.p_axis.plunge), _sf(pax.p_axis.azimuth)
        mt = getattr(fm, "moment_tensor", None)
        ten = getattr(mt, "tensor", None) if mt else None
        if ten:
            Mrr=_sf(ten.m_rr); Mtt=_sf(ten.m_tt); Mpp=_sf(ten.m_pp)
            Mrt=_sf(ten.m_rt); Mrp=_sf(ten.m_rp); Mtp=_sf(ten.m_tp)

    if (None in (T_pl,T_az,N_pl,N_az,P_pl,P_az)) and (None not in (s1,d1,r1)):
        T_pl,T_az,N_pl,N_az,P_pl,P_az = _axes_from_sdr(s1,d1,r1)
    if (None in (s1,d1,r1,s2,d2,r2)) and (None not in (Mrr,Mtt,Mpp,Mrt,Mrp,Mtp)):
        (s1,d1,r1),(s2,d2,r2) = _sdr_from_tensor_rtp(Mrr,Mtt,Mpp,Mrt,Mrp,Mtp)

    have_planes = None not in (s1, d1, r1) or None not in (s2, d2, r2)
    have_tensor = None not in (Mrr, Mtt, Mpp, Mrt, Mrp, Mtp)
    have_axes   = None not in (T_pl, T_az, N_pl, N_az, P_pl, P_az)
    if not (have_planes or have_tensor or have_axes):
        return None

    rid = str(getattr(ev.resource_id, "id", "")) or str(row["id"])
    eid = rid.split("/")[-1].split(":")[-1] or str(row["id"])
    source = (getattr(getattr(ev,"creation_info",None),"agency_id",None) or row.get("net") or "us")

    return dict(
        id=eid, time_iso=time_iso, longitude=lon, latitude=lat, depth=depth,
        mag=mag, mag_type=mag_type,
        strike1=s1, dip1=d1, rake1=r1, strike2=s2, dip2=d2, rake2=r2,
        T_plunge=T_pl, T_azimuth=T_az, N_plunge=N_pl, N_azimuth=N_az, P_plunge=P_pl, P_azimuth=P_az,
        Mrr=Mrr, Mtt=Mtt, Mpp=Mpp, Mrt=Mrt, Mrp=Mrp, Mtp=Mtp,
        source=str(source).lower()
    )

def build_csv_from_catalog(input_catalog: pd.DataFrame, out_path: str):
    rows, failed = [], []
    total = len(input_catalog)
    for i, row in input_catalog.iterrows():
        eid = str(row["id"])
        print(f"[ANSS] parsing {i+1}/{total} id: {eid}")
        blob = get_quakeml_bytes(row)
        if not blob:
            failed.append(eid); continue
        rec = parse_quakeml_row(blob, row)
        if rec:
            rows.append(rec)
        else:
            failed.append(eid)
        time.sleep(0.05)  # gentle pacing

    df = pd.DataFrame(rows, columns=FIELDS)
    if not df.empty:
        df = filter_df(df)  # <-- unified filter
        df.to_csv(out_path, index=False)
    print(f"[ANSS] wrote {out_path} with {len(df)} rows (from {total} ids)")

def prepare_anss(in_path: str, out_path: str):
    inputcat = pd.read_csv(in_path)
    # expect an 'id' column for ComCat fetch
    build_csv_from_catalog(inputcat, out_path)

# =========================================================
# ---------------------  ISC-GEM  -------------------------
# =========================================================
def _has_sdr(s1, d1, r1):
    return all(v is not None for v in (s1, d1, r1)) and all(_finite(v) for v in (s1, d1, r1))

def _has_tensor(Mrr, Mtt, Mpp, Mrt, Mrp, Mtp):
    vals = (Mrr, Mtt, Mpp, Mrt, Mrp, Mtp)
    if not all(_finite(v) for v in vals): return False
    return any(abs(float(v)) > 1e-12 for v in vals)

def _axes_from_sdr_gem(strike, dip, rake):
    st, di, ra = np.radians([strike, dip, rake])
    n = np.array([-np.sin(di)*np.sin(st),  np.sin(di)*np.cos(st), -np.cos(di)])
    s = np.array([ np.cos(ra)*np.cos(st)+np.sin(ra)*np.cos(di)*np.sin(st),
                   np.cos(ra)*np.sin(st)-np.sin(ra)*np.cos(di)*np.cos(st),
                   np.sin(ra)*np.sin(di)])
    n/=np.linalg.norm(n); s/=np.linalg.norm(s)
    M = np.outer(s,n) + np.outer(n,s)
    w, V = np.linalg.eigh(M); P, N, T = V[:,0], V[:,1], V[:,2]
    def azpl(v):
        v = v/np.linalg.norm(v)
        if v[2] < 0: v = -v
        az = (math.degrees(math.atan2(v[1], v[0])) + 360) % 360
        pl = math.degrees(math.asin(max(-1, min(1, v[2]))))
        return pl, az
    T_pl, T_az = azpl(T); N_pl, N_az = azpl(N); P_pl, P_az = azpl(P)
    return T_pl, T_az, N_pl, N_az, P_pl, P_az

def _sdr_from_tensor_rtp_gem(Mrr, Mtt, Mpp, Mrt, Mrp, Mtp):
    Mrtp = np.array([[Mrr, Mrt, Mrp],
                     [Mrt, Mtt, Mtp],
                     [Mrp, Mtp, Mpp]], float)
    A    = np.array([[0, -1, 0],
                     [0,  0, 1],
                     [-1, 0, 0]], float)
    M    = A @ Mrtp @ A.T
    w, V = np.linalg.eigh(M); idx = np.argsort(w)
    P, N, T = V[:, idx[0]], V[:, idx[1]], V[:, idx[2]]
    def up(v): return v if v[2] >= 0 else -v
    T, P = up(T), up(P)
    n1 = T + P; s1 = T - P
    n2 = T - P; s2 = T + P
    def n_s_to_sdr(n, s):
        n = n/np.linalg.norm(n); s = s/np.linalg.norm(s)
        if n[2] > 0: n = -n; s = -s
        dip = math.degrees(math.atan2(abs(n[2]), math.hypot(n[0], n[1])))
        strike = (math.degrees(math.atan2(n[1], n[0])) + 90) % 360
        phi = math.radians(strike)
        u_strike = np.array([-math.sin(phi), math.cos(phi), 0.0])
        delta    = math.radians(dip)
        u_dip    = np.array([math.cos(phi)*math.sin(delta), math.sin(phi)*math.sin(delta), -math.cos(delta)])
        rake = math.degrees(math.atan2(float(np.dot(s, u_dip)), float(np.dot(s, u_strike))))
        if rake > 180: rake -= 360
        if rake <= -180: rake += 360
        return strike, dip, rake
    return n_s_to_sdr(n1, s1), n_s_to_sdr(n2, s2)

def prepare_isc_gem(in_path: str, out_path: str):
    df = pd.read_csv(in_path, skipinitialspace=True)

    # map columns
    cols = {c.lower().strip(): c for c in df.columns}
    def col(*names):
        for k in names:
            if k.lower() in cols:
                return cols[k.lower()]
        return None

    c_date  = col("date","time","datetime")
    c_lat   = col("lat","latitude")
    c_lon   = col("lon","longitude")
    c_depth = col("depth","dep")
    c_mw    = col("mw","mag")
    c_id    = col("eventid","id","name")

    # tensor
    c_mpp = col("mpp"); c_mpr = col("mpr"); c_mrr = col("mrr")
    c_mrt = col("mrt"); c_mtp = col("mtp"); c_mtt = col("mtt")

    # SDR
    c_s1 = col("str1","strike1","s1","strike_1")
    c_d1 = col("dip1","d1","dip_1")
    c_r1 = col("rake1","r1","rake_1")
    c_s2 = col("str2","strike2","s2","strike_2")
    c_d2 = col("dip2","d2","dip_2")
    c_r2 = col("rake2","r2","rake_2")

    out_rows = []
    for _, r in df.iterrows():
        lat = _sf(r.get(c_lat))  if c_lat  else None
        lon = _sf(r.get(c_lon))  if c_lon  else None
        if (lat is None) or (lon is None):
            continue
        # spatial prefilter quick (keeps code lighter)
        if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
            continue

        time_iso = None
        if c_date and pd.notna(r.get(c_date)):
            t = pd.to_datetime(r[c_date], utc=True, errors="coerce")
            if pd.notna(t):
                time_iso = t.tz_convert(None).isoformat(timespec="seconds")

        depth = _sf(r.get(c_depth)) if c_depth else None
        mw    = _sf(r.get(c_mw))    if c_mw    else None
        eid   = str(r.get(c_id)).strip() if c_id and pd.notna(r.get(c_id)) else None

        s1 = _sf(r.get(c_s1)) if c_s1 else None
        d1 = _sf(r.get(c_d1)) if c_d1 else None
        k1 = _sf(r.get(c_r1)) if c_r1 else None
        s2 = _sf(r.get(c_s2)) if c_s2 else None
        d2 = _sf(r.get(c_d2)) if c_d2 else None
        k2 = _sf(r.get(c_r2)) if c_r2 else None

        Mpp = _sf(r.get(c_mpp)) if c_mpp else None
        Mrp = _sf(r.get(c_mpr)) if c_mpr else None
        Mrr = _sf(r.get(c_mrr)) if c_mrr else None
        Mrt = _sf(r.get(c_mrt)) if c_mrt else None
        Mtp = _sf(r.get(c_mtp)) if c_mtp else None
        Mtt = _sf(r.get(c_mtt)) if c_mtt else None

        have_sdr    = _has_sdr(s1, d1, k1)
        have_tensor = _has_tensor(Mrr, Mtt, Mpp, Mrt, Mrp, Mtp)
        if not (have_sdr or have_tensor):
            continue

        if (not have_sdr) and have_tensor:
            try:
                (s1, d1, k1), (s2, d2, k2) = _sdr_from_tensor_rtp_gem(Mrr, Mtt, Mpp, Mrt, Mrp, Mtp)
                have_sdr = True
            except Exception:
                pass

        T_pl = T_az = N_pl = N_az = P_pl = P_az = None
        if have_sdr:
            try:
                T_pl, T_az, N_pl, N_az, P_pl, P_az = _axes_from_sdr_gem(s1, d1, k1)
            except Exception:
                pass

        out_rows.append(dict(
            id=eid, time_iso=time_iso, longitude=lon, latitude=lat, depth=depth,
            mag=mw, mag_type=("mw" if mw is not None else None),
            strike1=s1, dip1=d1, rake1=k1,
            strike2=s2, dip2=d2, rake2=k2,
            T_plunge=T_pl, T_azimuth=T_az, N_plunge=N_pl, N_azimuth=N_az, P_plunge=P_pl, P_azimuth=P_az,
            Mrr=Mrr, Mtt=Mtt, Mpp=Mpp, Mrt=Mrt, Mrp=Mrp, Mtp=Mtp,
            source="gem"
        ))

    out = pd.DataFrame(out_rows, columns=FIELDS)
    out = filter_df(out)  # <-- unified filter
    out.to_csv(out_path, index=False)
    print(f"[ISC-GEM] wrote {out_path} with {len(out)} rows.")

# =========================================================
# -----------------------  GCMT  --------------------------
# =========================================================
Row = Dict[str, object]

def is_number(s: str) -> bool:
    try:
        float(s); return True
    except Exception:
        return False

def safe_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None

def _normalize_hms_for_datetime(time_s: str):
    m = re.match(r"^(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?$", time_s.strip())
    if not m:
        raise ValueError(f"Unrecognized time string: {time_s!r}")
    hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
    frac = float(f"0.{m.group(4)}") if m.group(4) is not None else 0.0
    if ss >= 60:
        ss, frac = 59, 0.0
    return hh, mm, ss, frac

def parse_datetime_ymd_slash(date_s: str, time_s: str) -> datetime:
    y, m, d = (int(x) for x in date_s.strip().split("/"))
    hh, mm, ss, frac = _normalize_hms_for_datetime(time_s)
    return datetime(y, m, d, hh, mm, ss) + timedelta(seconds=frac)

def parse_ndk_file(path: str) -> List[Row]:
    rows: List[Row] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip() != ""]
    if len(lines) < 5:
        return rows
    lines = lines[: (len(lines) // 5) * 5]

    for i in range(0, len(lines), 5):
        l1, l2, l3, l4, l5 = lines[i:i+5]
        m1 = re.match(
            r"^\s*(\S+)\s+"
            r"(\d{4}/\d{2}/\d{2})\s+"
            r"(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+"
            r"([+-]?\d+(?:\.\d+)?)\s+"
            r"([+-]?\d+(?:\.\d+)?)\s+"
            r"([+-]?\d+(?:\.\d+)?)\s+"
            r"([+-]?\d+(?:\.\d+)?)\s+"
            r"([+-]?\d+(?:\.\d+)?)\b",
            l1
        )
        if not m1: continue

        date_s = m1.group(2)
        time_s = m1.group(3)
        hdr_lat   = float(m1.group(4))
        hdr_lon   = float(m1.group(5))
        hdr_depth = float(m1.group(6))
        mb_val    = safe_float(m1.group(7))
        mw_hdr    = safe_float(m1.group(8))

        try:
            origin_time = parse_datetime_ymd_slash(date_s, time_s)
        except Exception:
            continue

        ev_id = l2.split()[0]

        try:
            after = l3.split("CENTROID:", 1)[1]
            nums = [safe_float(x) for x in after.split() if is_number(x)]
            tshift = float(nums[0])
            clat   = float(nums[2])
            clon   = float(nums[4])
            cdepth = float(nums[6])
        except Exception:
            tshift = 0.0
            clat, clon, cdepth = hdr_lat, hdr_lon, hdr_depth

        centroid_time = origin_time + timedelta(seconds=tshift)

        Mrr = Mtt = Mpp = Mrt = Mrp = Mtp = None
        try:
            nums4 = re.findall(r'[+-]?\d+(?:\.\d+)?', l4)
            if len(nums4) >= 13:
                exp = int(float(nums4[0]))
                mvals = [float(nums4[i]) for i in (1, 3, 5, 7, 9, 11)]
                factor = 10 ** (exp - 7)
                Mrr, Mtt, Mpp, Mrt, Mrp, Mtp = [v * factor for v in mvals]
        except Exception:
            pass

        nums5 = re.findall(r'(?:(?<=\s)|^)[+-]?\d+(?:\.\d+)?', l5)
        if len(nums5) < 16:
            nums5 = re.findall(r'[+-]?\d+(?:\.\d+)?', l5)
            if len(nums5) == 17 and nums5[0] in ("10", "11", "12"):
                nums5 = nums5[1:]
        if len(nums5) < 16:
            continue

        vals = list(map(float, nums5[:16]))
        T_plunge, T_azimuth = vals[1], vals[2]
        N_plunge, N_azimuth = vals[4], vals[5]
        P_plunge, P_azimuth = vals[7], vals[8]
        strike1, dip1, rake1 = vals[10], vals[11], vals[12]
        strike2, dip2, rake2 = vals[13], vals[14], vals[15]
        Mw = mw_hdr if (mw_hdr is not None and mw_hdr > 0) else mb_val
        mag_type = "mw"

        rows.append(dict(
            id=ev_id,
            time_iso=centroid_time.isoformat(timespec="seconds"),
            longitude=clon,
            latitude=clat,
            depth=cdepth,
            mag=float(Mw) if Mw is not None else None,
            mag_type=mag_type,
            strike1=strike1, dip1=dip1, rake1=rake1,
            strike2=strike2, dip2=dip2, rake2=rake2,
            T_plunge=T_plunge, T_azimuth=T_azimuth,
            N_plunge=N_plunge, N_azimuth=N_azimuth,
            P_plunge=P_plunge, P_azimuth=P_azimuth,
            Mrr=Mrr, Mtt=Mtt, Mpp=Mpp, Mrt=Mrt, Mrp=Mrp, Mtp=Mtp,
            source="gcmt",
        ))
    return rows

def prepare_gcmt(in_path1: str, in_path2: str, out_path: str):
    rows = []
    rows += parse_ndk_file(in_path1)
    rows += parse_ndk_file(in_path2)
    df = pd.DataFrame(rows, columns=FIELDS)
    df = filter_df(df)  # <-- unified filter (spatial + time >= 1976 + Mw>=4.95)
    df.to_csv(out_path, index=False)
    print(f"[GCMT] wrote {out_path} with {len(df)} rows.")

# =========================================================
# -------------------  GMT (Nico)  ------------------------
# =========================================================
_GMT_TIME_RE = re.compile(r"^(\d{12})[A-Za-z]$")  # YYYYMMDDhhmm + trailing letter (post-2005 IDs)

def _gmt_id_to_time_iso(gmt_id: str) -> Optional[str]:
    s = str(gmt_id or "").strip()
    m = _GMT_TIME_RE.match(s)
    if not m:
        return None
    digits = m.group(1)
    yyyy = int(digits[0:4]); mm = int(digits[4:6]); dd = int(digits[6:8])
    hh   = int(digits[8:10]); mi = int(digits[10:12])
    return f"{yyyy:04d}-{mm:02d}-{dd:02d}T{hh:02d}:{mi:02d}:00"

def prepare_gmt_nico(in_path: str, out_path: str) -> pd.DataFrame:
    raw = pd.read_csv(in_path)
    out = pd.DataFrame({
        "id":        raw.get("ID"),
        "time_iso":  raw["ID"].apply(_gmt_id_to_time_iso),  # may be None pre-2005
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
    out = ensure_fields_df(out)
    out = filter_df(out)  # <-- unified filter; with STRICT_TIME_FILTER=False, rows w/ missing time are kept
    out.to_csv(out_path, index=False)
    print(f"[GMT] wrote {out_path} with {len(out)} rows.")
    return out

# =========================================================
# ------------------------- MAIN --------------------------
# =========================================================
if __name__ == '__main__':

    OUT_DIR = 'processed_catalogs'
    os.makedirs(OUT_DIR, exist_ok=True)

    isc_priority = {"GFZ": 0, "GEOFON": 0, "NEIC": 1, "NEIS": 1, "US": 1, "USGS": 1}
    isc_in_path = 'raw_catalogs/isc.txt'
    isc_out_path = f'{OUT_DIR}/isc_formatted.csv'
    prepare_isc(isc_in_path, isc_out_path, isc_priority)

    # anss_in_path = 'raw_catalogs/anss.csv'
    # anss_out_path = f'{OUT_DIR}/anss_formatted.csv'
    # prepare_anss(anss_in_path, anss_out_path)

    isc_gem_in_path = "raw_catalogs/isc-gem-cat.csv"
    isc_gem_out_path = f'{OUT_DIR}/isc_gem_formatted.csv'
    prepare_isc_gem(isc_gem_in_path, isc_gem_out_path)

    gcmt_in_path1 = "raw_catalogs/gcmt_jan76_dec20.txt"
    gcmt_in_path2 = "raw_catalogs/GCMT_bundle_2025-08-29T16.59.01.txt"
    gcmt_out_path = f'{OUT_DIR}/gcmt_formatted.csv'
    prepare_gcmt(gcmt_in_path1, gcmt_in_path2, gcmt_out_path)

    gmt_nico_in_path = 'raw_catalogs/GMT_1976_2025_consolidado_conSlab.csv'
    gmt_nico_out_path = f'{OUT_DIR}/gmt_nico_formatted.csv'
    prepare_gmt_nico(gmt_nico_in_path, gmt_nico_out_path)
