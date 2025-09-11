#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Must be set BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from obspy.imaging.beachball import beachball as bb

# ---------- CONFIG ----------
INPUT_CSV   = "catalog_labeled_v2.csv"
OUT_DIR     = "beachballs_svg"  # folder will be created
BB_SIZE_PT  = 220               # beachball width (points)
MAX_WORKERS = None              # None -> use os.cpu_count()

CLASSES = [
    "crustal_intraarc_shallow",
    "crustal_intraarc_deep",
    "subduction_interface",
    "subduction_intraslab",
    "deep_subduction",
    "outer_rise",
    "forearc",
    "deep",
    "unclassified",
]

CLASS_COLORS = {
    "crustal_intraarc_shallow": "limegreen",
    "crustal_intraarc_deep":    "darkgreen",
    "subduction_interface":     "deepskyblue",
    "subduction_intraslab":     "teal",
    "deep_subduction":          "red",
    "outer_rise":               "burlywood",
    "forearc":                  "orange",
    "deep":                     "firebrick",
    "unclassified":             "#7f7f7f",
}
DEFAULT_COLOR = "#7f7f7f"

# ---------- helpers ----------
def _finite(x):
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def has_tensor(row) -> bool:
    """
    True only if all 6 components are finite AND not all zero.
    (So a 0/0/0/0/0/0 tensor falls back to nodal planes.)
    """
    comps = []
    for k in ("Mrr", "Mtt", "Mpp", "Mrt", "Mrp", "Mtp"):
        v = row.get(k)
        try:
            f = float(v)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(f):
            return False
        comps.append(f)
    EPS = 1e-12
    return any(abs(f) > EPS for f in comps)

def has_sdr(row) -> bool:
    return all(_finite(row.get(k)) for k in ("strike1","dip1","rake1"))

def get_class(row) -> str:
    # robust: find column named 'class' ignoring case
    for k in row.keys():
        if str(k).strip().lower() == "class":
            return str(row[k]).strip()
    return "unclassified"

def class_color(label: str) -> str:
    return CLASS_COLORS.get(label, DEFAULT_COLOR)

def draw_one_png(out_path: str, facecolor: str, mt=None, sdr=None, width_pt=220):
    # Dedicated figure per event
    fig = plt.figure(figsize=(width_pt/72.0, width_pt/72.0), dpi=72)
    fig.patch.set_alpha(0.0)  # transparent figure background

    if mt is not None:
        bb(mt, width=width_pt, facecolor=facecolor, edgecolor="black",
           linewidth=0.8, bgcolor="w", fig=fig)
    else:
        bb(tuple(sdr), width=width_pt, facecolor=facecolor, edgecolor="black",
           linewidth=0.8, bgcolor="w", fig=fig)

    for ax in fig.axes:
        ax.set_facecolor("none")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, format="png", dpi=30,
                bbox_inches="tight", pad_inches=0.0, transparent=True)
    plt.close(fig)

# ---------- worker (must be top-level for multiprocessing) ----------
def _render_worker(row_dict):
    """
    Returns ('mt'|'sdr'|'skip'|'err', event_id, msg_if_err_or_reason)
    """
    try:
        # Normalize event id
        eid_raw = row_dict.get("id")
        eid = "" if eid_raw is None else str(eid_raw).strip()
        if not eid or eid.lower() == "nan":
            return ("skip", eid, "no id")

        label = get_class(row_dict)
        color = class_color(label)
        out_path = os.path.join(OUT_DIR, f"{eid}.png")

        if has_tensor(row_dict):
            mt = [float(row_dict["Mrr"]), float(row_dict["Mtt"]), float(row_dict["Mpp"]),
                  float(row_dict["Mrt"]), float(row_dict["Mrp"]), float(row_dict["Mtp"])]
            draw_one_png(out_path, color, mt=mt, sdr=None, width_pt=BB_SIZE_PT)
            return ("mt", eid, "")
        elif has_sdr(row_dict):
            sdr = (float(row_dict["strike1"]), float(row_dict["dip1"]), float(row_dict["rake1"]))
            draw_one_png(out_path, color, mt=None, sdr=sdr, width_pt=BB_SIZE_PT)
            return ("sdr", eid, "")
        else:
            return ("skip", eid, "no MT/SDR")
    except Exception as e:
        # Return error but don't raise (so pool keeps going)
        return ("err", str(row_dict.get("id", "")), str(e))

# ---------- main ----------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    if "id" not in df.columns:
        raise SystemExit("CSV must have an 'id' column.")
    rows = df.to_dict("records")

    ok_mt = ok_sdr = skipped = errs = 0

    # Use processes for Matplotlib safety
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        # chunksize speeds up map for many small tasks
        futures = [ex.submit(_render_worker, r) for r in rows]
        for fut in as_completed(futures):
            kind, eid, msg = fut.result()
            if kind == "mt":
                ok_mt += 1
            elif kind == "sdr":
                ok_sdr += 1
            elif kind == "skip":
                skipped += 1
            else:
                errs += 1
                # optional: print or log a few errors
                # print(f"[err] {eid}: {msg}")

    total = len(rows)
    print(f"Done. Rendered MT: {ok_mt}, SDR: {ok_sdr}, skipped: {skipped}, errors: {errs}, total rows: {total}")
    print(f"PNGs written to: {OUT_DIR}")

if __name__ == "__main__":
    main()
