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

from cat_handler import paths
from cat_handler.parsers.tools import _clean_token, _to_iso, _sf, _to_nm, filter_df

LAT_MIN, LAT_MAX = -51.0, -27.0
LON_MIN, LON_MAX = -80.0, -65.0
TIME_MIN = "1976-01-01T00:00:00"
MAG_MIN = 4.95
DROP_NO_TIMES = False

TIME_MIN = pd.to_datetime(TIME_MIN, utc=True)

FIELDS = [
    "id","time_iso","longitude","latitude","depth","mag","mag_type",
    "strike1","dip1","rake1","strike2","dip2","rake2",
    "T_plunge","T_azimuth","N_plunge","N_azimuth","P_plunge","P_azimuth",
    "Mrr","Mtt","Mpp","Mrt","Mrp","Mtp","source",
]


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
    df = filter_df(df,
                   lon_min=LON_MIN, lon_max=LON_MAX,
                   lat_min=LAT_MIN, lat_max=LAT_MAX,
                   time_min=TIME_MIN,
                   mag_min=MAG_MIN,
                   drop_notimes=DROP_NO_TIMES)
    df.to_csv(out_path, index=False)
    print(f"[ISC] wrote {out_path}  ({len(df)} rows)")



if __name__ == "__main__":

    isc_priority = {"GFZ": 0, "GEOFON": 0, "NEIC": 1, "NEIS": 1, "US": 1, "USGS": 1}
    prepare_isc(paths.rawcat_isc, paths.cat_isc, isc_priority)