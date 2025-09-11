#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd

# -------------------- OUTPUT SCHEMA --------------------
FIELDS = [
    "id","time_iso","longitude","latitude","depth","mag","mag_type",
    "strike1","dip1","rake1","strike2","dip2","rake2",
    "T_plunge","T_azimuth","N_plunge","N_azimuth","P_plunge","P_azimuth",
    "Mrr","Mtt","Mpp","Mrt","Mrp","Mtp","source",
]

# Spatial filter (strict inequalities as requested)
LAT_MIN, LAT_MAX = -54.0, -17.0
LON_MIN, LON_MAX = -80.0, -65.0

# --------------- helpers ---------------
def _sf(x):
    """safe float -> float or None"""
    try:
        f = float(x)
        return f if np.isfinite(f) else None
    except Exception:
        return None

def _finite(x):
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def _has_sdr(s1, d1, r1):
    return all(v is not None for v in (s1, d1, r1)) and all(_finite(v) for v in (s1, d1, r1))

def _has_tensor(Mrr, Mtt, Mpp, Mrt, Mrp, Mtp):
    vals = (Mrr, Mtt, Mpp, Mrt, Mrp, Mtp)
    if not all(_finite(v) for v in vals):
        return False
    # treat all-zeros as “no tensor”
    return any(abs(float(v)) > 1e-12 for v in vals)

def _axes_from_sdr(strike, dip, rake):
    """
    Compute T/N/P (plunge, azimuth) from one nodal plane (strike,dip,rake).
    Returns (T_pl, T_az, N_pl, N_az, P_pl, P_az).
    """
    st, di, ra = np.radians([strike, dip, rake])
    # Fault-normal n and slip direction s (Aki & Richards)
    n = np.array([-np.sin(di)*np.sin(st),  np.sin(di)*np.cos(st), -np.cos(di)])
    s = np.array([ np.cos(ra)*np.cos(st)+np.sin(ra)*np.cos(di)*np.sin(st),
                   np.cos(ra)*np.sin(st)-np.sin(ra)*np.cos(di)*np.cos(st),
                   np.sin(ra)*np.sin(di)])
    n /= np.linalg.norm(n); s /= np.linalg.norm(s)
    M = np.outer(s,n) + np.outer(n,s)  # unit DC tensor in NED
    w, V = np.linalg.eigh(M); P, N, T = V[:,0], V[:,1], V[:,2]

    def azpl(v):
        v = v/np.linalg.norm(v)
        if v[2] < 0: v = -v
        az = (math.degrees(math.atan2(v[1], v[0])) + 360) % 360
        pl = math.degrees(math.asin(max(-1, min(1, v[2]))))
        return pl, az

    T_pl, T_az = azpl(T); N_pl, N_az = azpl(N); P_pl, P_az = azpl(P)
    return T_pl, T_az, N_pl, N_az, P_pl, P_az

def _sdr_from_tensor_rtp(Mrr, Mtt, Mpp, Mrt, Mrp, Mtp):
    """
    Best-DC nodal planes from a full moment tensor given in RTP (r=up, t=south, p=east).
    Returns ((strike1,dip1,rake1), (strike2,dip2,rake2)) in degrees.
    """
    # RTP -> NED transform
    Mrtp = np.array([[Mrr, Mrt, Mrp],
                     [Mrt, Mtt, Mtp],
                     [Mrp, Mtp, Mpp]], float)
    A    = np.array([[0, -1, 0],
                     [0,  0, 1],
                     [-1, 0, 0]], float)
    M    = A @ Mrtp @ A.T  # NED

    # eigenvectors -> P,N,T
    w, V = np.linalg.eigh(M); idx = np.argsort(w)
    P, N, T = V[:, idx[0]], V[:, idx[1]], V[:, idx[2]]

    # Build candidate planes from T & P (Kagan, standard recipe)
    def _azpl_to_vec(az, pl):
        az, pl = np.radians([az, pl])
        return np.array([np.cos(pl)*np.cos(az), np.cos(pl)*np.sin(az), np.sin(pl)])

    def _vec_to_azpl(v):
        v = v/np.linalg.norm(v)
        az = (math.degrees(math.atan2(v[1], v[0])) + 360) % 360
        pl = math.degrees(math.asin(max(-1, min(1, v[2]))))
        return az, pl

    # Convert eigenvectors to az/pl then back to unit vectors pointing up hemisphere
    def up(v):
        return v if v[2] >= 0 else -v
    T = up(T); P = up(P)

    # Candidate slip/normal (unit)
    # n = normalized (T_hat + P_hat), s = normalized (T_hat - P_hat)
    n1 = T + P; s1 = T - P
    n2 = T - P; s2 = T + P

    def n_s_to_sdr(n, s):
        n = n/np.linalg.norm(n); s = s/np.linalg.norm(s)
        # enforce downward normal for dip in [0,90]
        if n[2] > 0:
            n = -n; s = -s
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

# --------------- main conversion ---------------
def convert_gem_to_standard(in_path: str, out_path: str):
    """
    Read GEM CSV with (among others):
      date,lat,lon,depth,mw, mpp,mpr,mrr,mrt,mtp,mtt, str1,dip1,rake1,str2,dip2,rake2, eventid
    Filter to -54<lat<-17 and -80<lon<-65.
    Keep rows that have SDR or Tensor (drop rows with neither).
    Fill T/N/P from SDR; if missing SDR but tensor exists, derive SDR from tensor then T/N/P.
    Write standard schema CSV.
    """
    df = pd.read_csv(in_path, skipinitialspace=True)

    # normalize column names
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

    # GEM tensor names
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
        # coords + spatial filter
        lat = _sf(r.get(c_lat))  if c_lat  else None
        lon = _sf(r.get(c_lon))  if c_lon  else None
        if (lat is None) or (lon is None):
            continue
        if not (LAT_MIN < lat < LAT_MAX and LON_MIN < lon < LON_MAX):
            continue

        # time
        time_iso = None
        if c_date and pd.notna(r.get(c_date)):
            t = pd.to_datetime(r[c_date], utc=True, errors="coerce")
            if pd.notna(t):
                time_iso = t.tz_convert(None).isoformat(timespec="seconds")

        depth = _sf(r.get(c_depth)) if c_depth else None
        mw    = _sf(r.get(c_mw))    if c_mw    else None
        eid   = str(r.get(c_id)).strip() if c_id and pd.notna(r.get(c_id)) else None

        # SDR from file
        s1 = _sf(r.get(c_s1)) if c_s1 else None
        d1 = _sf(r.get(c_d1)) if c_d1 else None
        k1 = _sf(r.get(c_r1)) if c_r1 else None
        s2 = _sf(r.get(c_s2)) if c_s2 else None
        d2 = _sf(r.get(c_d2)) if c_d2 else None
        k2 = _sf(r.get(c_r2)) if c_r2 else None

        # Tensor mapping (GEM -> our names)
        Mpp = _sf(r.get(c_mpp)) if c_mpp else None
        Mrp = _sf(r.get(c_mpr)) if c_mpr else None
        Mrr = _sf(r.get(c_mrr)) if c_mrr else None
        Mrt = _sf(r.get(c_mrt)) if c_mrt else None
        Mtp = _sf(r.get(c_mtp)) if c_mtp else None
        Mtt = _sf(r.get(c_mtt)) if c_mtt else None

        have_sdr    = _has_sdr(s1, d1, k1)
        have_tensor = _has_tensor(Mrr, Mtt, Mpp, Mrt, Mrp, Mtp)

        # Drop rows with neither SDR nor tensor
        if not (have_sdr or have_tensor):
            continue

        # If SDR missing but tensor exists -> derive best-DC SDR from tensor
        if (not have_sdr) and have_tensor:
            try:
                (s1, d1, k1), (s2, d2, k2) = _sdr_from_tensor_rtp(Mrr, Mtt, Mpp, Mrt, Mrp, Mtp)
                have_sdr = True
            except Exception:
                pass

        # T/N/P from SDR if we have it
        T_pl = T_az = N_pl = N_az = P_pl = P_az = None
        if have_sdr:
            try:
                T_pl, T_az, N_pl, N_az, P_pl, P_az = _axes_from_sdr(s1, d1, k1)
            except Exception:
                pass

        out_rows.append(dict(
            id=eid,
            time_iso=time_iso,
            longitude=lon,
            latitude=lat,
            depth=depth,
            mag=mw,
            mag_type="mw" if mw is not None else None,
            strike1=s1, dip1=d1, rake1=k1,
            strike2=s2, dip2=d2, rake2=k2,
            T_plunge=T_pl, T_azimuth=T_az,
            N_plunge=N_pl, N_azimuth=N_az,
            P_plunge=P_pl, P_azimuth=P_az,
            Mrr=Mrr, Mtt=Mtt, Mpp=Mpp, Mrt=Mrt, Mrp=Mrp, Mtp=Mtp,
            source="gem",
        ))

    out = pd.DataFrame(out_rows, columns=FIELDS)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(out)} rows (filtered to lat/lon window).")

if __name__ == "__main__":
    # example:
    convert_gem_to_standard("./global/isc-gem-cat.csv", "gem_standard.csv")
