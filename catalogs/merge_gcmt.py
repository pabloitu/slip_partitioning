import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

Row = Dict[str, object]

# --- helpers you already had ---
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
    if ss >= 60:  # clamp leap-ish 60 to 59.0
        ss, frac = 59, 0.0
    return hh, mm, ss, frac

def parse_datetime_ymd_slash(date_s: str, time_s: str) -> datetime:
    y, m, d = (int(x) for x in date_s.strip().split("/"))
    hh, mm, ss, frac = _normalize_hms_for_datetime(time_s)
    return datetime(y, m, d, hh, mm, ss) + timedelta(seconds=frac)

# -------- unified NDK parser (works for both of your files) --------
def parse_ndk_file(path: str) -> List[Row]:
    """
    NDK: 5 lines per event.
    L1: 'SRC yyyy/mm/dd HH:MM:SS.s lat lon depth Mb Mw ...'
    L2: event id (first token)
    L3: 'CENTROID: tshift terr  clat claterr  clon clonerr  cdepth cdeptherr ...'
    L4: exponent + 6 tensor elements (ignored here)
    L5: 'Vxx  (T_eig T_pl T_az) (N_eig N_pl N_az) (P_eig P_pl P_az)  M0  s1 d1 r1  s2 d2 r2'
        We extract T/N/P (plunge, azimuth) and both nodal planes (strike,dip,rake).
    """
    rows: List[Row] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip() != ""]

    if len(lines) < 5:
        return rows
    lines = lines[: (len(lines) // 5) * 5]

    for i in range(0, len(lines), 5):
        l1, l2, l3, l4, l5 = lines[i : i + 5]

        # --- L1: header ---
        m1 = re.match(
            r"^\s*(\S+)\s+"                              # source tag
            r"(\d{4}/\d{2}/\d{2})\s+"                    # date
            r"(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+"          # time
            r"([+-]?\d+(?:\.\d+)?)\s+"                   # lat
            r"([+-]?\d+(?:\.\d+)?)\s+"                   # lon
            r"([+-]?\d+(?:\.\d+)?)\s+"                   # depth (km)
            r"([+-]?\d+(?:\.\d+)?)\s+"                   # Mb
            r"([+-]?\d+(?:\.\d+)?)\b",                   # Mw
            l1,
        )
        if not m1:
            continue

        src       = "gcmt"
        date_s    = m1.group(2)
        time_s    = m1.group(3)
        hdr_lat   = float(m1.group(4))
        hdr_lon   = float(m1.group(5))
        hdr_depth = float(m1.group(6))
        mb_val    = safe_float(m1.group(7))
        mw_hdr    = safe_float(m1.group(8))

        try:
            origin_time = parse_datetime_ymd_slash(date_s, time_s)
        except Exception:
            continue

        # --- L2: event id (first token) ---
        ev_id = l2.split()[0]

        # --- L3: centroid ---
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
        # --- L4: exponent + 6 tensor elements (values + uncertainties) ---
        # Format: <exp>  Mrr  dMrr  Mtt  dMtt  Mpp  dMpp  Mrt  dMrt  Mrp  dMrp  Mtp  dMtp
        # Units in dyne·cm × 10**exp  (convert to N·m by × 10**(exp-7))
        Mrr = Mtt = Mpp = Mrt = Mrp = Mtp = None
        try:
            nums4 = re.findall(r'[+-]?\d+(?:\.\d+)?', l4)
            if len(nums4) >= 13:
                exp = int(float(nums4[0]))  # exponent for dyne·cm
                # take the 6 values (skip 6 uncertainties)
                mvals = [float(nums4[i]) for i in (1, 3, 5, 7, 9, 11)]
                factor = 10 ** (exp - 7)    # dyne·cm -> N·m
                Mrr, Mtt, Mpp, Mrt, Mrp, Mtp = [v * factor for v in mvals]
        except Exception:
            pass
        # --- L5: principal axes + best-DC nodal planes ---
        # Numbers only if preceded by whitespace/start to avoid grabbing the '10' from 'V10'
        nums5 = re.findall(r'(?:(?<=\s)|^)[+-]?\d+(?:\.\d+)?', l5)
        # Expect: 9 (T_eig T_pl T_az N_eig N_pl N_az P_eig P_pl P_az) + 1 (M0) + 6 (sdr1,sdr2) = 16
        if len(nums5) < 16:
            # try a more permissive fallback (and drop a leading '10' if captured from 'V10')
            nums5 = re.findall(r'[+-]?\d+(?:\.\d+)?', l5)
            if len(nums5) == 17 and nums5[0] in ("10", "11", "12"):
                nums5 = nums5[1:]
        if len(nums5) < 16:
            # can't parse, skip event
            continue

        vals = list(map(float, nums5[:16]))
        # Map assuming order: T, N, P triples
        T_plunge, T_azimuth = vals[1], vals[2]
        N_plunge, N_azimuth = vals[4], vals[5]
        P_plunge, P_azimuth = vals[7], vals[8]
        # scalar moment = vals[9] (ignored)
        strike1, dip1, rake1 = vals[10], vals[11], vals[12]
        strike2, dip2, rake2 = vals[13], vals[14], vals[15]

        # Magnitude rule
        Mw = mw_hdr if (mw_hdr is not None and mw_hdr > 0) else mb_val
        mag_type = "Mw"
        print(ev_id, clon, clat, Mw)
        rows.append(dict(
            id=ev_id,
            time_iso=centroid_time.isoformat(timespec="seconds"),
            longitude=clon,
            latitude=clat,
            depth=cdepth,
            mag=float(Mw) if Mw is not None else None,
            mag_type=mag_type,
            # Nodal planes
            strike1=strike1, dip1=dip1, rake1=rake1,
            strike2=strike2, dip2=dip2, rake2=rake2,
            # Principal axes (degrees)
            T_plunge=T_plunge, T_azimuth=T_azimuth,
            N_plunge=N_plunge, N_azimuth=N_azimuth,
            P_plunge=P_plunge, P_azimuth=P_azimuth,
            # Moment tensor (RTP) in N·m
            Mrr=Mrr, Mtt=Mtt, Mpp=Mpp, Mrt=Mrt, Mrp=Mrp, Mtp=Mtp,
            source="gcmt",
        ))

    return rows


if __name__ == "__main__":
    # === INPUTS (both are NDK files now) ===
    old_cat = "./global_2/gcmt_jan76_dec20.txt"
    new_cat = "./global_2/GCMT_bundle_2025-08-29T16.59.01.txt"

    # Filters (set to None to disable)
    MINLAT, MAXLAT = -51, -27
    MINLON, MAXLON = -80, -65
    MINDEPTH, MAXDEPTH = 0, 150
    MINMW, MAXMW = 4.9, 9.9

    OUT_CSV = "./global_2/merged_gcmt_mechanisms.csv"

    # Parse
    rows = []
    rows += parse_ndk_file(old_cat)
    rows += parse_ndk_file(new_cat)

    # Filter
    def in_bounds(r):
        lat, lon, dep, mw = r["latitude"], r["longitude"], r["depth"], r["mag"]
        if lat is None or lon is None: return False
        if MINLAT is not None and lat < MINLAT: return False
        if MAXLAT is not None and lat > MAXLAT: return False
        if MINLON is not None and lon < MINLON: return False
        if MAXLON is not None and lon > MAXLON: return False
        if MINDEPTH is not None and dep is not None and dep < MINDEPTH: return False
        if MAXDEPTH is not None and dep is not None and dep > MAXDEPTH: return False
        if MINMW is not None and mw is not None and mw < MINMW: return False
        if MAXMW is not None and mw is not None and mw > MAXMW: return False
        return True

    rows = [r for r in rows if in_bounds(r)]

    # (Optional) deduplicate by id/time/loc if your two files overlap
    # You already had a deduplicate() helper; you can keep using it if you want. :contentReference[oaicite:1]{index=1}
    # rows = deduplicate(rows)

    # Sort & write
    rows.sort(key=lambda r: r["time_iso"])
    fieldnames = [
        "id", "time_iso", "longitude", "latitude", "depth", "mag", "mag_type",
        "strike1", "dip1", "rake1", "strike2", "dip2", "rake2",
        "T_plunge", "T_azimuth", "N_plunge", "N_azimuth", "P_plunge", "P_azimuth",
        "Mrr", "Mtt", "Mpp", "Mrt", "Mrp", "Mtp", "source"
    ]

    import csv
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

