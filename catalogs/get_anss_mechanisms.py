# build_anss_mechanisms.py
# Requirements: requests, numpy
# Output: an NDK-like mechanisms CSV matching your GCMT file fields.

import csv
import math
import requests
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import numpy as np

USGS_EVENT_DETAIL_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

IN_CSV  = "global_2/anss.csv"                 # <-- your ANSS CSV (as shown)
OUT_CSV = "anss_mechanisms.csv"               # <-- output like GCMT mechanisms


# ----------------------- helpers -----------------------
def _sf(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _best_mag(mag: str, magtype: str) -> Optional[float]:
    """
    Prefer Mw-type mags from the ANSS CSV; else just use the given magnitude.
    """
    if mag is None or mag == "":
        return None
    mag = float(mag)
    if magtype:
        mt = magtype.strip().lower()
        if mt.startswith("mw"):  # mww, mwc, mwb, mwr, ...
            return mag
    return mag


def _pick_product(prod_list: List[Dict], prefer_source: Optional[str] = "us") -> Optional[Dict]:
    """
    Choose a product from a products list:
    - prefer the requested source network (e.g., 'us'),
    - else pick the one flagged 'preferred',
    - else first in list.
    """
    if not prod_list:
        return None
    if prefer_source:
        for p in prod_list:
            if p.get("source") == prefer_source:
                return p
    for p in prod_list:
        if p.get("preferred"):
            return p
    return prod_list[0]


def _get_products(event_id: str) -> Dict[str, List[Dict]]:
    """
    Fetch ComCat detail for a single event id; return the 'products' dict.
    (We ask for includesuperseded=true to widen the set, but we still choose
    a single 'preferred' one per product type.)
    """
    params = {
        "format": "geojson",
        "eventid": event_id,
        "includesuperseded": "true",
    }
    r = requests.get(USGS_EVENT_DETAIL_URL, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    feats = js.get("features") or []
    if not feats:
        raise ValueError(f"No features for eventid={event_id}")
    props = feats[0].get("properties") or {}
    return props.get("products") or {}


# ---- compute T/N/P from SDR if needed (pure DC) ----
def _sdr_to_axes(strike: float, dip: float, rake: float) -> Tuple[Tuple[float,float], Tuple[float,float], Tuple[float,float]]:
    """
    Convert (strike,dip,rake) to T/N/P axes (azimuth, plunge) in degrees.
    We build a DC moment tensor from SDR (in NED), then eigendecompose.
    """
    # Convert to radians
    st, di, ra = map(np.radians, [strike, dip, rake])

    # Fault normal (upper hemisphere) and slip vector in NED
    # Following Aki & Richards conventions:
    # strike clockwise from North; dip 0..90; rake -180..180 (positive down-dip)
    # Unit vectors:
    #   n (plane normal), s (slip within plane)
    # Build direction cosines
    # Reference: standard focal mech conversions
    nN = -np.sin(di) * np.sin(st)
    nE =  np.sin(di) * np.cos(st)
    nD = -np.cos(di)
    sN =  np.cos(ra) * np.cos(st) + np.sin(ra) * np.cos(di) * np.sin(st)
    sE =  np.cos(ra) * np.sin(st) - np.sin(ra) * np.cos(di) * np.cos(st)
    sD =  np.sin(ra) * np.sin(di)

    n = np.array([nN, nE, nD]); n /= np.linalg.norm(n)
    s = np.array([sN, sE, sD]); s /= np.linalg.norm(s)

    # DC moment tensor (unit moment is fine for axes)
    # M = (s ⊗ n + n ⊗ s) (symmetric)
    M = np.outer(s, n) + np.outer(n, s)

    # Eigen-decomp (ascending)
    w, V = np.linalg.eigh(M)
    idx = np.argsort(w)
    V = V[:, idx]  # columns: P (min), N (mid), T (max)

    P = V[:, 0]; N = V[:, 1]; T = V[:, 2]

    def vec_to_az_pl(v):
        v = v / np.linalg.norm(v)
        # ensure downward plunge (D >= 0)
        if v[2] < 0:
            v = -v
        Nn, Ee, Dd = v
        az = (np.degrees(np.arctan2(Ee, Nn)) + 360.0) % 360.0
        pl = np.degrees(np.arcsin(np.clip(Dd, -1.0, 1.0)))
        return az, pl

    T_az, T_pl = vec_to_az_pl(T)
    N_az, N_pl = vec_to_az_pl(N)
    P_az, P_pl = vec_to_az_pl(P)

    return (T_az, T_pl), (N_az, N_pl), (P_az, P_pl)


def extract_mechanism_for_event(row: Dict[str, str]) -> Optional[Dict[str, object]]:
    """
    Given one CSV row from your ANSS file, query ComCat and assemble:
      id,time_iso,lon,lat,depth_km,Mw,strike1,dip1,rake1,strike2,dip2,rake2,T_plunge,T_azimuth,N_plunge,N_azimuth,P_plunge,P_azimuth,source
    """
    eid = row.get("id")
    if not eid:
        return None

    products = _get_products(eid)

    # FOCAL-MECHANISM: nodal planes (most direct)
    fm = _pick_product(products.get("focal-mechanism", []), prefer_source="us")
    fm_props = fm.get("properties") if fm else {}

    s1 = _sf(fm_props.get("nodal-plane-1-strike"))
    d1 = _sf(fm_props.get("nodal-plane-1-dip"))
    r1 = _sf(fm_props.get("nodal-plane-1-rake"))
    s2 = _sf(fm_props.get("nodal-plane-2-strike"))
    d2 = _sf(fm_props.get("nodal-plane-2-dip"))
    r2 = _sf(fm_props.get("nodal-plane-2-rake"))

    # MOMENT-TENSOR: T/N/P and sometimes nodal planes too
    mt = _pick_product(products.get("moment-tensor", []), prefer_source="us")
    mt_props = mt.get("properties") if mt else {}

    t_pl = _sf(mt_props.get("t-axis-plunge"));  t_az = _sf(mt_props.get("t-axis-azimuth"))
    n_pl = _sf(mt_props.get("n-axis-plunge"));  n_az = _sf(mt_props.get("n-axis-azimuth"))
    p_pl = _sf(mt_props.get("p-axis-plunge"));  p_az = _sf(mt_props.get("p-axis-azimuth"))

    # If nodal planes missing but MT has them (sometimes included) use those:
    if s1 is None:
        s1 = _sf(mt_props.get("nodal-plane-1-strike"))
        d1 = _sf(mt_props.get("nodal-plane-1-dip"))
        # some feeds use 'slip' instead of 'rake'
        r1 = _sf(mt_props.get("nodal-plane-1-rake") or mt_props.get("nodal-plane-1-slip"))
    if s2 is None:
        s2 = _sf(mt_props.get("nodal-plane-2-strike"))
        d2 = _sf(mt_props.get("nodal-plane-2-dip"))
        r2 = _sf(mt_props.get("nodal-plane-2-rake") or mt_props.get("nodal-plane-2-slip"))

    # If axes missing but we have SDR, compute axes
    if (t_pl is None or t_az is None or n_pl is None or n_az is None or p_pl is None or p_az is None) and \
       (s1 is not None and d1 is not None and r1 is not None):
        (t_az, t_pl), (n_az, n_pl), (p_az, p_pl) = _sdr_to_axes(s1, d1, r1)

    # Time/loc: prefer MT-derived centroid if available; else use CSV hypocenter
    time_iso = row.get("time")  # CSV already ISO (e.g., 2025-08-22T20:42:25.416Z)
    lon = _sf(mt_props.get("derived-longitude")) or _sf(row.get("longitude"))
    lat = _sf(mt_props.get("derived-latitude"))  or _sf(row.get("latitude"))
    dep = _sf(mt_props.get("derived-depth"))     or _sf(row.get("depth"))
    # derived-eventtime is ISO 8601 per docs
    det = mt_props.get("derived-eventtime")
    if det:
        time_iso = det

    # Magnitude rule
    Mw = _best_mag(row.get("mag"), row.get("magType"))

    # If still missing any essential pieces, skip
    if not ( (s1 is not None and d1 is not None and r1 is not None) and
             (s2 is not None and d2 is not None and r2 is not None) and
             (t_pl is not None and t_az is not None and
              n_pl is not None and n_az is not None and
              p_pl is not None and p_az is not None) ):
        # We require both planes and axes to match your GCMT-style output
        return None

    source = None
    if fm and fm.get("source"):
        source = fm["source"]
    elif mt and mt.get("source"):
        source = mt["source"]
    else:
        source = row.get("net")  # fallback

    return dict(
        id=eid,
        time_iso=time_iso.replace("Z", "+00:00") if time_iso else None,
        longitude=lon, latitude=lat, depth_km=dep, Mw=Mw,
        strike1=s1, dip1=d1, rake1=r1, strike2=s2, dip2=d2, rake2=r2,
        T_plunge=t_pl, T_azimuth=t_az,
        N_plunge=n_pl, N_azimuth=n_az,
        P_plunge=p_pl, P_azimuth=p_az,
        source=source,
    )


# ----------------------- run over CSV -----------------------
def read_anss_csv(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def write_mech_csv(rows: List[Dict[str, object]], out_path: str):
    fieldnames = [
        "id","time_iso","longitude","latitude","depth_km","Mw",
        "strike1","dip1","rake1","strike2","dip2","rake2",
        "T_plunge","T_azimuth","N_plunge","N_azimuth","P_plunge","P_azimuth","source"
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    in_rows = read_anss_csv(IN_CSV)
    out_rows = []
    misses = []
    for i, r in enumerate(in_rows, 1):
        eid = r.get("id")
        try:
            mech = extract_mechanism_for_event(r)
            if mech:
                out_rows.append(mech)
            else:
                misses.append(eid)
        except Exception as e:
            misses.append(f"{eid} ({e})")
        if i % 25 == 0:
            print(f"Processed {i}/{len(in_rows)}…")

    # sort by time
    out_rows.sort(key=lambda x: (x["time_iso"] or "", x["id"]))
    write_mech_csv(out_rows, OUT_CSV)
    print(f"wrote {OUT_CSV} with {len(out_rows)} events")
    if misses:
        print(f" {len(misses)} events missing mechanism/axes (see ComCat pages):")
        for m in misses[:20]:
            print("   -", m)
        if len(misses) > 20:
            print("   …")
