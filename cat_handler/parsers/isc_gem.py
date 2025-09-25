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

from cat_handler import paths
from cat_handler.parsers.tools import _sf, _has_sdr, _has_tensor, _finite, filter_df


LAT_MIN, LAT_MAX = -57.0, -17.0
LON_MIN, LON_MAX = -80.0, -65.0
TIME_MIN_STR = "1976-01-01T00:00:00"
MAG_MIN = 4.95

TIME_MIN = pd.to_datetime(TIME_MIN_STR, utc=True)

FIELDS = [
    "id","time_iso","longitude","latitude","depth","mag","mag_type",
    "strike1","dip1","rake1","strike2","dip2","rake2",
    "T_plunge","T_azimuth","N_plunge","N_azimuth","P_plunge","P_azimuth",
    "Mrr","Mtt","Mpp","Mrt","Mrp","Mtp","source",
]

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
    out = filter_df(out, lon_min=LON_MIN, lon_max=LON_MAX, lat_min=LAT_MIN, lat_max=LAT_MAX, mag_min=MAG_MIN,
                   time_min=TIME_MIN)
    out.to_csv(out_path, index=False)
    print(f"[ISC-GEM] wrote {out_path} with {len(out)} rows.")

if __name__ == "__main__":


    prepare_isc_gem(paths.rawcat_isc_gem, './cat_gem_chile.csv')