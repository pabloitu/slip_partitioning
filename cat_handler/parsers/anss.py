from __future__ import annotations

import io
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from libcomcat.search import get_event_by_id
from obspy import read_events

from cat_handler import paths
from cat_handler.parsers.tools import (
    _is_quakeml_bytes,
    _maybe_decompress,
    _pause,
    _preferred_origin,
    _sf,
    _sdr_from_tensor_rtp,
    _to_bytes,
    filter_df,
)

# -------------------- CONFIG --------------------
LAT_MIN, LAT_MAX = -51.0, -27.0
LON_MIN, LON_MAX = -80.0, -65.0
TIME_MIN = pd.to_datetime("1976-01-01T00:00:00", utc=True)
MAG_MIN = 4.95
DROP_NO_TIMES = False

# -------------------- OUTPUT SCHEMA --------------------
FIELDS = [
    "id",
    "time_iso",
    "longitude",
    "latitude",
    "depth",
    "mag",
    "mag_type",
    "lon_error",
    "lat_error",
    "depth_error",
    "mag_error",
    "strike1",
    "dip1",
    "rake1",
    "strike2",
    "dip2",
    "rake2",
    "Mrr",
    "Mtt",
    "Mpp",
    "Mrt",
    "Mrp",
    "Mtp",
    "source",
]

# -------------------- MANUAL ENTRIES (if product fetch fails) --------------------
UNQUERYABLE_EARTHQUAKES: List[Dict[str, Any]] = [
    dict(
        id="choy19850303224726",
        time_iso="1985-03-03T22:47:26",
        longitude=-71.62,
        latitude=-33.12,
        depth=33.0,
        mag=8.0,
        mag_type="mw",
        lon_error=None,
        lat_error=None,
        depth_error=None,
        mag_error=None,
        strike1=360.0,
        dip1=35.0,
        rake1=105.0,
        strike2=360.0,
        dip2=35.0,
        rake2=105.0,
        Mrr=None,
        Mtt=None,
        Mpp=None,
        Mrt=None,
        Mrp=None,
        Mtp=None,
        source="anss",
    ),
    dict(
        id="official20100227063411530_30",
        time_iso="2010-02-27T06:34:11",
        longitude=-72.898,
        latitude=-36.122,
        depth=30.0,
        mag=8.8,
        mag_type="mww",
        lon_error=None,
        lat_error=None,
        depth_error=None,
        mag_error=None,
        strike1=178.0,
        dip1=77.0,
        rake1=86.0,
        strike2=17.0,
        dip2=14.0,
        rake2=108.0,
        Mrr=None,
        Mtt=None,
        Mpp=None,
        Mrt=None,
        Mrp=None,
        Mtp=None,
        source="anss",
    ),
]

# -------------------- FETCH PRODUCTS --------------------
def get_quakeml_bytes(
    row: pd.Series,
    min_interval: float = 0.35,
    tries_event: int = 2,
    tries_content: int = 2,
    backoff_base: float = 2.0,
) -> bytes | None:
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
            time.sleep((backoff_base**k) + random.uniform(0, 0.5))
    if ev is None:
        return None

    def _safe_get(_ev, name):
        try:
            return _ev.getProducts(name) or []
        except Exception:
            return []

    def _collect(ptypes):
        out = []
        for ptype in ptypes:
            prods = _safe_get(ev, ptype)
            if not prods:
                continue
            us = [p for p in prods if getattr(p, "source", None) == "anss"]
            out += (us if us else prods)
        return out

    candidates = _collect(("focal-mechanism", "moment-tensor")) or _collect(("phase-data", "origin"))
    if not candidates:
        return None

    def _candidate_names(prod):
        names = []
        try:
            for key in getattr(prod, "contents", {}) or {}:
                lk = key.lower()
                if ("quakeml" in lk) and (lk.endswith(".xml") or lk.endswith(".xml.gz") or lk.endswith(".zip")):
                    names.append(key)
            for key in getattr(prod, "contents", {}) or {}:
                lk = key.lower()
                if lk.endswith(".xml") or lk.endswith(".xml.gz"):
                    names.append(key)
        except Exception:
            pass
        seen, ordered = set(), []
        for n in names:
            if n not in seen:
                seen.add(n)
                ordered.append(n)
        ordered += ["quakeml.xml", "quakeml"]
        return ordered

    import zipfile

    for prod in candidates:
        names = _candidate_names(prod)
        for k in range(tries_content):
            for name in names:
                try:
                    raw = prod.getContentBytes(name)
                    b = _to_bytes(raw)
                    if not b:
                        continue
                    lk = name.lower()
                    if lk.endswith(".xml.gz"):
                        b = _maybe_decompress(b)
                    elif lk.endswith(".zip"):
                        try:
                            with zipfile.ZipFile(io.BytesIO(b)) as zf:
                                members = [n for n in zf.namelist() if n.lower().endswith(".xml")]
                                if not members:
                                    continue
                                b = zf.read(members[0])
                        except Exception:
                            continue
                    if b and _is_quakeml_bytes(b):
                        return b
                except Exception:
                    continue
            if k < tries_content - 1:
                time.sleep((backoff_base**k) + random.uniform(0, 0.5))

    print(f"[{eid}] products present but no usable QuakeML (xml/xml.gz/zip)")
    return None

# -------------------- PARSE ONE ROW --------------------
def parse_quakeml_row(xml_bytes: bytes, row: pd.Series) -> dict | None:
    cat = read_events(io.BytesIO(xml_bytes))
    if not cat:
        return None
    ev = cat[0]
    org = _preferred_origin(ev)

    # Time
    time_iso: Optional[str] = None
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

    # Location & depth
    lon = _sf(getattr(org, "longitude", None)) if org and getattr(org, "longitude", None) is not None else _sf(row.get("longitude"))
    lat = _sf(getattr(org, "latitude", None)) if org and getattr(org, "latitude", None) is not None else _sf(row.get("latitude"))
    if org and getattr(org, "depth", None) is not None:
        depth = _sf(org.depth) / 1000.0
    else:
        depth = _sf(row.get("depth"))

    # Magnitude (prefer Mw-like from QuakeML)
    pref_order = ["mww", "mwc", "mwr", "mwb", "mw"]
    mag = None
    mag_type = None
    best_rank = 10**9
    for m in (getattr(ev, "magnitudes", None) or []):
        t = (getattr(m, "magnitude_type", None) or "").strip().lower()
        if not t.startswith("mw") or m.mag is None:
            continue
        rank = pref_order.index(t) if t in pref_order else pref_order.index("mw")
        if rank < best_rank:
            best_rank = rank
            mag = float(m.mag)
            mag_type = t
    if mag is None:
        mag = _sf(row.get("mag"))
        mt = row.get("magType") or row.get("mag_type")
        mag_type = (str(mt).strip().lower() if mt is not None else None)

    # Focal mechanism / tensor
    s1 = d1 = r1 = s2 = d2 = r2 = None
    Mrr = Mtt = Mpp = Mrt = Mrp = Mtp = None

    fm = ev.focal_mechanisms[0] if ev.focal_mechanisms else None
    if fm:
        npn = getattr(fm, "nodal_planes", None)
        if npn and npn.nodal_plane_1 and npn.nodal_plane_2:
            s1, d1, r1 = float(npn.nodal_plane_1.strike), float(npn.nodal_plane_1.dip), float(npn.nodal_plane_1.rake)
            s2, d2, r2 = float(npn.nodal_plane_2.strike), float(npn.nodal_plane_2.dip), float(npn.nodal_plane_2.rake)
        mt = getattr(fm, "moment_tensor", None)
        ten = getattr(mt, "tensor", None) if mt else None
        if ten:
            Mrr = _sf(ten.m_rr)
            Mtt = _sf(ten.m_tt)
            Mpp = _sf(ten.m_pp)
            Mrt = _sf(ten.m_rt)
            Mrp = _sf(ten.m_rp)
            Mtp = _sf(ten.m_tp)

    # If SDR missing but tensor exists, derive SDR from tensor
    if (None in (s1, d1, r1, s2, d2, r2)) and (None not in (Mrr, Mtt, Mpp, Mrt, Mrp, Mtp)):
        (s1, d1, r1), (s2, d2, r2) = _sdr_from_tensor_rtp(Mrr, Mtt, Mpp, Mrt, Mrp, Mtp)

    # Quick sanity: if we have none of planes/tensor, skip
    have_planes = None not in (s1, d1, r1) or None not in (s2, d2, r2)
    have_tensor = None not in (Mrr, Mtt, Mpp, Mrt, Mrp, Mtp)
    if not (have_planes or have_tensor):
        return None

    rid = str(getattr(ev.resource_id, "id", "")) or str(row["id"])
    eid = rid.split("/")[-1].split(":")[-1] or str(row["id"])
    source = "anss"

    # From ANSS/ComCat CSV:
    # - horizontalError -> set lon_error and lat_error equal (same horizontal 1-sigma)
    # - depthError      -> depth_error
    # - magError        -> mag_error
    h_err = _sf(row.get("horizontalError"))
    lon_error = h_err
    lat_error = h_err
    depth_error = _sf(row.get("depthError"))
    mag_error = _sf(row.get("magError"))

    return dict(
        id=eid,
        time_iso=time_iso,
        longitude=lon,
        latitude=lat,
        depth=depth,
        mag=mag,
        mag_type=mag_type,
        lon_error=lon_error,
        lat_error=lat_error,
        depth_error=depth_error,
        mag_error=mag_error,
        strike1=s1,
        dip1=d1,
        rake1=r1,
        strike2=s2,
        dip2=d2,
        rake2=r2,
        Mrr=Mrr,
        Mtt=Mtt,
        Mpp=Mpp,
        Mrt=Mrt,
        Mrp=Mrp,
        Mtp=Mtp,
        source="anss",
    )

# -------------------- DRIVER --------------------
def build_csv_from_catalog(
    input_catalog: pd.DataFrame,
    out_path: str | bytes | None,
    manual_rows: Optional[List[Dict[str, Any]]] = None,
) -> None:
    rows: List[Dict[str, Any]] = []
    failed: List[str] = []
    total = len(input_catalog)

    for i, row in input_catalog.iterrows():
        eid = str(row["id"])
        print(f"[ANSS] parsing {i+1}/{total} id: {eid}")
        blob = get_quakeml_bytes(row)
        if not blob:
            failed.append(eid)
            continue
        rec = parse_quakeml_row(blob, row)
        if rec:
            rows.append(rec)
        else:
            failed.append(eid)
        time.sleep(0.05)

    df = pd.DataFrame(rows, columns=FIELDS)

    if not df.empty:
        df = filter_df(
            df,
            lon_min=LON_MIN,
            lon_max=LON_MAX,
            lat_min=LAT_MIN,
            lat_max=LAT_MAX,
            mag_min=MAG_MIN,
            time_min=TIME_MIN,
        )

    if manual_rows:
        manual_df = pd.DataFrame(manual_rows, columns=FIELDS)
        if not manual_df.empty:
            manual_df = filter_df(
                manual_df,
                lon_min=LON_MIN,
                lon_max=LON_MAX,
                lat_min=LAT_MIN,
                lat_max=LAT_MAX,
                mag_min=MAG_MIN,
                time_min=TIME_MIN,
            )
        df = pd.concat([manual_df, df], ignore_index=True)
        df = df.drop_duplicates(subset=["id"], keep="first")

    if not df.empty and out_path:
        df.to_csv(out_path, index=False)

    print(f"[ANSS] wrote {out_path} with {len(df)} rows (from {total} ids; failed {len(failed)})")

def prepare_anss(in_path: str | bytes, out_path: str | bytes) -> None:
    inputcat = pd.read_csv(in_path)
    build_csv_from_catalog(inputcat, out_path, manual_rows=UNQUERYABLE_EARTHQUAKES)

if __name__ == "__main__":
    prepare_anss(paths.rawcat_anss, paths.cat_anss)
