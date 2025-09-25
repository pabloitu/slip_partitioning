
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

STRICT_TIME_FILTER = False
_last_request_ts = 0.0

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
    return val * (10.0 ** exponent) * 1e-7

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


def filter_df(df: pd.DataFrame,
              lon_min=-180,
              lon_max=180,
              lat_min=-90,
              lat_max=90,
              mag_min=4.95,
              time_min="1976-01-01T00:00:00",
              drop_notimes: bool = False) -> pd.DataFrame:

    df = df.copy()

    # Numeric coercion for spatial + mag
    for col in ["longitude", "latitude", "mag"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Spatial mask
    m_spatial = (
        df["latitude"].between(lat_min, lat_max) &
        df["longitude"].between(lon_min, lon_max)
    )

    # Magnitude mask (require finite mag and >= MAG_MIN)
    m_mag = pd.to_numeric(df["mag"], errors="coerce").ge(mag_min)

    # Time mask
    if "time_iso" in df.columns:
        t = pd.to_datetime(df["time_iso"], utc=True, errors="coerce")
        if drop_notimes:
            m_time = t.ge(time_min)
        else:
            m_time = t.ge(time_min) | t.isna()
    else:
        # no time column -> keep all if not strict; drop all if strict
        m_time = (~drop_notimes)

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

