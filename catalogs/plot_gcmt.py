# plot_beachball.py
# pip install obspy matplotlib

import csv
from obspy.imaging.beachball import beachball
import matplotlib.pyplot as plt

MECH_CSV = "merged_gcmt_mechanisms.csv"
EVENT_ID = "C202503182117A"   # <-- set the ID you want to plot
PLANE    = 2# 1 or 2

def load_by_id_from_mech(event_id, path=MECH_CSV):
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("id") == event_id:
                return r
    return None

def plot_for_id(event_id, plane=1, size=250, path=MECH_CSV):
    m = load_by_id_from_mech(event_id, path=path)
    if not m:
        raise SystemExit(f"Event {event_id} not found in {path}")

    # choose nodal plane
    if plane == 2:
        sdr = (float(m["strike2"]), float(m["dip2"]), float(m["rake2"]))
        plane_tag = "plane 2"
    else:
        sdr = (float(m["strike1"]), float(m["dip1"]), float(m["rake1"]))
        plane_tag = "plane 1"

    # title bits (optional fields guarded)
    mw   = m.get("Mw")
    dep  = m.get("depth_km")
    time = m.get("time_iso", "")
    mw_s  = f"M={float(mw):.1f}" if mw not in (None, "", "None") else ""
    dep_s = f"z={float(dep):.0f} km" if dep not in (None, "", "None") else ""
    pieces = [event_id, plane_tag, mw_s, dep_s, time]
    title = "  •  ".join([p for p in pieces if p])

    beachball(sdr, size=size, linewidth=1)  # draw from SDR


if __name__ == "__main__":
    plot_for_id(EVENT_ID, plane=PLANE)
