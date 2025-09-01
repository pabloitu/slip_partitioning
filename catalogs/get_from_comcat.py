# compact_libcomcat_to_csv.py
# deps: pip install usgs-libcomcat obspy numpy pandas

import io, math
import numpy as np
import pandas as pd
from obspy import read_events
from libcomcat.search import get_event_by_id
from concurrent.futures import ThreadPoolExecutor, as_completed
import time, gzip
FIELDS = [
    "id","time_iso","longitude","latitude","depth","mag","mag_type",
    "strike1","dip1","rake1","strike2","dip2","rake2",
    "T_plunge","T_azimuth","N_plunge","N_azimuth","P_plunge","P_azimuth",
    "Mrr","Mtt","Mpp","Mrt","Mrp","Mtp","source"
]
import time, random, gzip

_last_request_ts = 0.0  # module-level, used for pacing

def _pause(min_interval: float):
    global _last_request_ts
    now = time.monotonic()
    dt = now - _last_request_ts
    if dt < min_interval:
        time.sleep(min_interval - dt)
    _last_request_ts = time.monotonic()

def _is_quakeml_bytes(b: bytes) -> bool:
    if not b:
        return False
    head = b[:4096].lower()
    return (b"<?xml" in head or b"<q:quakeml" in head or b"<quakeml" in head)

def _maybe_decompress(b: bytes) -> bytes:
    if not b:
        return b
    if len(b) >= 2 and b[0] == 0x1F and b[1] == 0x8B:
        try:
            return gzip.decompress(b)
        except Exception:
            return b
    return b


def _process_one(row_dict):
    """Fetch + parse one event (safe for threads)."""
    try:
        blob = get_quakeml_bytes(row_dict)
        if not blob:
            return None
        rec = parse_quakeml_row(blob, row_dict)
        return rec
    except Exception:
        return None


def _sf(x):
    try: return float(x)
    except: return None

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
    # RTP->NED
    Mrtp = np.array([[mrr,mrt,mrp],[mrt,mtt,mtp],[mrp,mtp,mpp]], float)
    A    = np.array([[0,-1,0],[0,0,1],[-1,0,0]], float)
    Mned = A @ Mrtp @ A.T
    w,V  = np.linalg.eigh(Mned); V = V[:, np.argsort(w)]  # P,N,T
    P,N,T = V[:,0],V[:,1],V[:,2]
    def v(az,pl):
        az,pl=np.radians([az,pl]); return np.array([np.cos(pl)*np.cos(az),np.cos(pl)*np.sin(az),np.sin(pl)])
    # get az/pl for T,P
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
    # prefer MT-derived origin with a valid time; else first origin WITH time; else preferred_origin()
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
    # any origin with time
    for org in (ev.origins or []):
        if getattr(org, "time", None):
            return org
    # last resort
    return ev.preferred_origin() or (ev.origins[0] if (ev.origins or []) else None)


def get_quakeml_bytes(row,
                      min_interval: float = 0.35,
                      tries_event: int = 2,
                      tries_content: int = 2,
                      backoff_base: float = 2.0) -> bytes | None:
    """
    Fetch QuakeML bytes for an event row using libcomcat.
    - No retries if event has no FM/MT products (deterministic).
    - Retries ONLY on transient network/content errors.
    - Gentle pacing via min_interval seconds between calls.
    """
    eid = str(row["id"])
    # --- pace requests globally
    _pause(min_interval)

    # 1) get event (retry only if libcomcat fails to fetch JSON)
    ev = None
    for k in range(tries_event):
        try:
            ev = get_event_by_id(eventid=eid, includesuperseded=True)
            break
        except Exception as e:
            if k == tries_event - 1:
                print(f"[{eid}] get_event_by_id failed after {tries_event} tries: {e}")
                return None
            sleep_s = (backoff_base ** k) + random.uniform(0, 0.5)
            time.sleep(sleep_s)
    if ev is None:
        return None

    # 2) find FM/MT products; if neither exists, DO NOT retry
    fm_prods = []
    mt_prods = []
    try:
        fm_prods = ev.getProducts("focal-mechanism") or []
    except Exception:
        fm_prods = []
    try:
        mt_prods = ev.getProducts("moment-tensor") or []
    except Exception:
        mt_prods = []

    if not fm_prods and not mt_prods:
        # deterministic absence -> no retry
        # (You can log this if you want to re-check later with a different source)
        # print(f"[{eid}] no FM/MT products in ComCat")
        return None

    # prefer USGS source; else first
    def _pick(prods):
        if not prods: return []
        us = [p for p in prods if getattr(p, "source", None) == "us"]
        return us if us else prods

    candidates = _pick(fm_prods) + _pick(mt_prods)

    # 3) try to extract any quakeml*.xml(.gz) content; retry ONLY if we fail to download/parse bytes
    # scan available content names first (shortest names first)
    def _candidate_names(prod):
        names = []
        try:
            for key in getattr(prod, "contents", []):
                lk = key.lower()
                if ("quakeml" in lk and lk.endswith(".xml")) or lk.endswith(".xml.gz"):
                    names.append(key)
        except Exception:
            pass
        # Always try the common names first
        # Put 'quakeml.xml' (if present) in front
        names = sorted(set(["quakeml.xml", "quakeml"] + names), key=len)
        return names

    for prod in candidates:
        names = _candidate_names(prod)
        if not names:
            # fall back to the known names even if not advertised
            names = ["quakeml.xml", "quakeml"]

        for k in range(tries_content):
            for name in names:
                try:
                    blob = prod.getContentBytes(name)  # may be (bytes|str|tuple)
                    b = _to_bytes(blob)
                    b = _maybe_decompress(b)
                    if b and _is_quakeml_bytes(b):
                        return b
                except Exception:
                    # try next name
                    continue
            # none of the names succeeded -> backoff and try again (transient)
            if k < tries_content - 1:
                sleep_s = (backoff_base ** k) + random.uniform(0, 0.5)
                time.sleep(sleep_s)

    # reached here: products exist but we couldn't retrieve usable QuakeML bytes
    print(f"[{eid}] FM/MT present but no usable QuakeML content")
    return None

# ---------- parse with robust fallbacks to input row ----------
def parse_quakeml_row(xml_bytes: bytes, row) -> dict | None:
    cat = read_events(io.BytesIO(xml_bytes))
    if not cat:
        return None
    ev = cat[0]

    # Origin (robust)
    org = _preferred_origin(ev)

    # time
    time_iso = None
    if org and getattr(org, "time", None):
        try:
            time_iso = org.time.datetime.replace(tzinfo=None).isoformat(timespec="seconds")
        except Exception:
            time_iso = None
    if time_iso is None:
        # fallback to CSV time
        try:
            t = pd.to_datetime(row["time"], utc=True, errors="coerce")
            if pd.notna(t):
                time_iso = t.tz_convert(None).isoformat(timespec="seconds")
        except Exception:
            time_iso = None

    # lon/lat/depth
    lon = _sf(getattr(org, "longitude", None)) if org and getattr(org, "longitude", None) is not None else _sf(row.get("longitude"))
    lat = _sf(getattr(org, "latitude",  None)) if org and getattr(org, "latitude",  None) is not None else _sf(row.get("latitude"))
    depth = None
    if org and getattr(org, "depth", None) is not None:
        depth = _sf(org.depth) / 1000.0  # QuakeML meters -> km
    else:
        depth = _sf(row.get("depth"))    # CSV already in km

    # magnitude (prefer Mw from QuakeML; else CSV)
    mag = mag_type = None
    for m in (getattr(ev, "magnitudes", None) or []):
        t = (getattr(m, "magnitude_type", None) or "").upper()
        if t.startswith("MW") and m.mag is not None:
            mag, mag_type = float(m.mag), "Mw"; break
    if mag is None:
        mag, mag_type = _sf(row.get("mag")), (row.get("magType") or "Mw")

    # planes, axes, tensor
    s1, d1, r1, s2, d2, r2 = (None,) * 6
    T_pl, T_az, N_pl, N_az, P_pl, P_az = (None,) * 6
    Mrr, Mtt, Mpp, Mrt, Mrp, Mtp = (None,) * 6

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

    # compute missing axes from plane 1 if needed
    if (None in (T_pl,T_az,N_pl,N_az,P_pl,P_az)) and (None not in (s1,d1,r1)):
        T_pl,T_az,N_pl,N_az,P_pl,P_az = _axes_from_sdr(s1,d1,r1)
    # compute best-DC planes from tensor if planes missing but tensor exists
    if (None in (s1,d1,r1,s2,d2,r2)) and (None not in (Mrr,Mtt,Mpp,Mrt,Mrp,Mtp)):
        (s1,d1,r1),(s2,d2,r2) = _sdr_from_tensor_rtp(Mrr,Mtt,Mpp,Mrt,Mrp,Mtp)


    # New: keep the row if we have *any* mechanism info (planes OR tensor OR axes).
    have_planes = None not in (s1, d1, r1) or None not in (s2, d2, r2)
    have_tensor = None not in (Mrr, Mtt, Mpp, Mrt, Mrp, Mtp)
    have_axes   = None not in (T_pl, T_az, N_pl, N_az, P_pl, P_az)

    if not (have_planes or have_tensor or have_axes):
        # No mechanism info at all; skip
        return None

    # id & source
    rid = str(getattr(ev.resource_id, "id", "")) or str(row["id"])
    eid = rid.split("/")[-1].split(":")[-1] or str(row["id"])
    source = (getattr(getattr(ev,"creation_info",None),"agency_id",None) or row.get("net") or "us")

    return dict(
        id=eid, time_iso=time_iso, longitude=lon, latitude=lat, depth=depth,
        mag=mag, mag_type=mag_type,
        strike1=s1, dip1=d1, rake1=r1, strike2=s2, dip2=d2, rake2=r2,
        T_plunge=T_pl, T_azimuth=T_az, N_plunge=N_pl, N_azimuth=N_az, P_plunge=P_pl, P_azimuth=P_az,
        Mrr=Mrr, Mtt=Mtt, Mpp=Mpp, Mrt=Mrt, Mrp=Mrp, Mtp=Mtp,
        source=source
    )
def build_csv_from_catalog(input_catalog: pd.DataFrame, out_path: str, failed_log: str | None = None):
    rows, failed = [], []
    total = len(input_catalog)
    for i, row in input_catalog.iterrows():
        eid = str(row["id"])
        print(f"Parsing {i+1}/{total}  id: {eid}")
        blob = get_quakeml_bytes(row)
        if not blob:
            failed.append(eid)
            continue
        rec = parse_quakeml_row(blob, row)
        if rec:
            rows.append(rec)
        else:
            print(f"[{eid}] parsed but missing all mech info (skipped)")
            # optional: keep a stub instead of skipping entirely
            failed.append(eid)

        # gentle pacing helps avoid throttling (tweak or remove)
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            by="time_iso",
            key=lambda s: pd.to_datetime(s, utc=True, errors="coerce"),
            na_position="last",
        )
        df.to_csv(out_path, index=False, columns=FIELDS)
    print(f"wrote {out_path} with {len(rows)} rows (from {total} ids)")

    if failed_log and failed:
        with open(failed_log, "w", encoding="utf-8") as f:
            for eid in failed:
                f.write(f"{eid}\n")
        print(f"{len(failed)} failures logged to {failed_log}")



inputcat = pd.read_csv('./global_2/anss.csv')
build_csv_from_catalog(inputcat, './global_2/anss_mechanisms.csv')
