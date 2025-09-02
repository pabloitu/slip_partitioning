# parse_isc_fm.py
# Parse ISC focal mechanism table, prefer GFZ>NEIC>others (skip GCMT),
# and write CSV compatible with your GCMT/ComCat outputs.
#
# Usage: python parse_isc_fm.py input_isc.txt output_isc_mechanisms.csv

import sys
import math
from datetime import datetime
from typing import List, Dict, Any

FIELDS = [
    "id","time_iso","longitude","latitude","depth","mag","mag_type",
    "strike1","dip1","rake1","strike2","dip2","rake2",
    "T_plunge","T_azimuth","N_plunge","N_azimuth","P_plunge","P_azimuth",
    "Mrr","Mtt","Mpp","Mrt","Mrp","Mtp","source"
]

PREF_RANK = {
    # best
    "GFZ": 0, "GEOFON": 0,
    # next
    "NEIC": 1, "NEIS": 1, "US": 1, "USGS": 1,
    # others (default)
}

def _clean_token(s: str) -> str:
    """Drop URL part like 'GCMT <https://...>' -> 'GCMT'. Also trim."""
    s = s.strip()
    if "<" in s:
        s = s.split("<", 1)[0].strip()
    return s

def _sf(x: str):
    try:
        return float(x.strip())
    except Exception:
        return None

def _to_iso(date_s: str, time_s: str) -> str:
    # DATE 'YYYY-MM-DD', TIME 'HH:MM:SS(.ss)'
    s = f"{date_s.strip()} {time_s.strip()}"
    # allow fractional seconds:
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).isoformat(timespec="seconds")
        except ValueError:
            continue
    return None

def _to_nm(val: float | None, exponent: int | None) -> float | None:
    """Convert dyne-cm * 10**exponent -> N·m. (1 dyne·cm = 1e-7 N·m)"""
    if val is None or exponent is None:
        return None
    return val * (10.0 ** exponent) * 1e-7

def parse_isc_file(path: str) -> List[Dict[str, Any]]:
    """
    Return a list of records containing all sources (we’ll dedupe/choose later).
    """
    rows: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]

    # find the header row that starts with "EVENT_ID,"
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("EVENT_ID,"):
            header_idx = i
            break
    if header_idx is None:
        print("Header row not found. Aborting.")
        return rows

    header = [h.strip() for h in lines[header_idx].split(",")]
    # Build column index map
    idx = {name: header.index(name) for name in header}

    # Helpers to fetch a field by name safely
    def get(rowlist, name, default=""):
        try:
            return rowlist[idx[name]]
        except Exception:
            return default

    # Parse each subsequent non-empty line until a blank or a new section
    for ln in lines[header_idx + 1:]:
        if not ln.strip():
            continue
        # Some ISC pages may have trailing footers; stop when the shape changes radically
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < len(header):
            # likely end of table
            continue

        # Extract raw fields
        raw_event = get(parts, "EVENT_ID")
        if not raw_event:
            continue
        event_id = _clean_token(raw_event).split()[0]  # '718362 <link>' -> '718362'

        date_s = get(parts, "DATE")
        time_s = get(parts, "TIME")
        time_iso = _to_iso(date_s, time_s)

        lat = _sf(get(parts, "LAT"))
        lon = _sf(get(parts, "LON"))
        depth = _sf(get(parts, "DEPTH"))  # km

        # FM author (second AUTHOR column)
        fm_author = _clean_token(get(parts, "AUTHOR", ""))  # this will fetch the FIRST "AUTHOR"
        # But in this table there are two "AUTHOR" columns; the second (FM AUTHOR) is after CENTROID.
        # If both exist, the header literally contains two "AUTHOR" entries.
        # We need the *second* one. Handle that by locating all positions of "AUTHOR".
        author_positions = [i for i, h in enumerate(header) if h == "AUTHOR"]
        if len(author_positions) >= 2:
            fm_author = _clean_token(parts[author_positions[1]])
        fm_author_up = fm_author.upper()

        # Skip GCMT (you already have those)
        if fm_author_up == "GCMT":
            continue

        # Magnitude: MW
        mw_str = get(parts, "MW", "").strip()
        mag = _sf(mw_str)
        mag_type = "Mw" if mag is not None else None

        # Tensor components: exponent + values (note ISC order: MRR, MTT, MPP, MRT, MTP, MPR)
        ex_comp = None
        try:
            ex_comp = int(get(parts, "EX"))  # this EX is the one before the components
        except Exception:
            # If the first EX is MO exponent, and the second EX is components exponent,
            # header has EX twice; grab the second occurrence.
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
        Mrp = _to_nm(_sf(get(parts, "MPR")), ex_comp)  # note: ISC uses MPR (== Mrp)

        # Nodal planes
        s1 = _sf(get(parts, "STRIKE"))
        d1 = _sf(get(parts, "DIP"))
        r1 = _sf(get(parts, "RAKE"))
        # second plane columns follow immediately after the first three
        # find indexes of the two sets from header occurrences
        strike_positions = [i for i, h in enumerate(header) if h == "STRIKE"]
        dip_positions    = [i for i, h in enumerate(header) if h == "DIP"]
        rake_positions   = [i for i, h in enumerate(header) if h == "RAKE"]
        if len(strike_positions) >= 2 and len(dip_positions) >= 2 and len(rake_positions) >= 2:
            s2 = _sf(parts[strike_positions[1]])
            d2 = _sf(parts[dip_positions[1]])
            r2 = _sf(parts[rake_positions[1]])
        else:
            s2 = d2 = r2 = None

        # Principal axes (plunge, azimuth) — header lists EX, T_VAL, T_PL, T_AZM, P_VAL, P_PL, P_AZM, N_VAL, N_PL, N_AZM
        # We only keep the orientations.
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
            _rank=PREF_RANK.get(fm_author_up, 9)  # for later selection
        ))

    return rows

def choose_best_per_event(all_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group by id and pick best source by rank; if tie, prefer the one with most filled fields."""
    by_id: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_rows:
        by_id.setdefault(r["id"], []).append(r)

    out: List[Dict[str, Any]] = []
    for eid, group in by_id.items():
        # sort by rank (low is better)
        group.sort(key=lambda r: (r.get("_rank", 9), ))
        best_rank = group[0].get("_rank", 9)
        cands = [g for g in group if g.get("_rank", 9) == best_rank]

        def filled_score(rec: Dict[str, Any]) -> int:
            keys = ["strike1","dip1","rake1","strike2","dip2","rake2",
                    "T_plunge","T_azimuth","N_plunge","N_azimuth","P_plunge","P_azimuth",
                    "Mrr","Mtt","Mpp","Mrt","Mrp","Mtp"]
            return sum(1 for k in keys if rec.get(k) is not None)

        # pick the one with the most information; tie-breaker: most recent time_iso
        cands.sort(key=lambda r: (filled_score(r), r.get("time_iso") or ""), reverse=True)
        chosen = cands[0].copy()
        chosen.pop("_rank", None)
        out.append(chosen)

    # sort by time
    out.sort(key=lambda r: r.get("time_iso") or "")
    return out

def write_csv(rows: List[Dict[str, Any]], path: str):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            # ensure all required fields exist
            rec = {k: r.get(k, None) for k in FIELDS}
            w.writerow(rec)


if __name__ == '__main__':
    in_path = './global_2/isc.txt'
    out_path = './global_2/isc_mechanisms.txt'
    all_rows = parse_isc_file(in_path)
    if not all_rows:
        print("No rows parsed.")

    selected = choose_best_per_event(all_rows)
    write_csv(selected, out_path)
    print(f"Wrote {len(selected)} events to {out_path}")

