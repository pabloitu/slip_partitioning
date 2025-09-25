#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render beachball PNGs for selected classified catalogs (no CLI).

- Edit SOURCES_TO_RENDER in main() to choose which subfolders under
  'classified_catalogs' to render, e.g. ["merged"] or ["gcmt","anss"].
- Looks for a single overall "*_classified.csv" in each source folder
  (skips per-class CSVs named after class labels).
- Writes to: classified_catalogs/<source>/beachballs/<id>.png
- Parallelized with ProcessPoolExecutor.

NEW:
- ONLY_DOUBLE_COUPLE: if True, always render a DC beachball using SDR.
  If SDR is missing but a tensor exists, derive DC planes from the tensor
  with obspy.mt2plane and use PREFERRED_DC_PLANE (1 or 2).
"""

import os
import glob
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Must be set BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from obspy.imaging.beachball import beachball as bb
from obspy.imaging.beachball import mt2plane

# ---------- ROOT ----------
ROOT = "classified_catalogs"  # contains subfolders: gcmt, anss, isc, gmt, merged, full, ...

# ---------- BEACHBALL SETTINGS ----------
BB_SIZE_PT  = 220   # beachball width (points)
DPI_OUT     = 30    # low DPI (QGIS symbol usage)
FMT         = "png" # png with transparent outside
MAX_WORKERS = None  # None -> os.cpu_count()

# ---------- RENDER MODE ----------
ONLY_DOUBLE_COUPLE   = True  # <<< set True to ALWAYS draw DC (from SDR or best-DC from tensor)
PREFERRED_DC_PLANE   = 1      # 1 or 2 — which nodal plane to use when deriving DC from tensor

# ---------- CLASSES & COLORS ----------
CLASSES = [
    "crustal_intraarc_shallow",
    "crustal_intraarc_deep",
    "slab_interface",
    "intra_slab",
    "slab_deep",
    "outer_rise",
    "forearc",
    "backarc",
    "unclassified",
]
CLASS_COLORS = {
    "crustal_intraarc_shallow": "limegreen",
    "crustal_intraarc_deep":    "darkgreen",
    "slab_interface":           "deepskyblue",
    "intra_slab":               "teal",
    "outer_rise":               "burlywood",
    "forearc":                  "orange",
    "backarc":                  "mediumpurple",
    "slab_deep":                "firebrick",
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
    """All 6 components finite AND not all zero."""
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
    return any(abs(f) > 1e-12 for f in comps)

def has_sdr(row) -> bool:
    return all(_finite(row.get(k)) for k in ("strike1","dip1","rake1"))

def get_sdr_for_dc(row):
    """
    Return (strike,dip,rake) for DC drawing:
      - Use provided SDR if available,
      - else derive best-DC planes from tensor and return plane #PREFERRED_DC_PLANE,
      - else return None.
    """
    if has_sdr(row):
        return (float(row["strike1"]), float(row["dip1"]), float(row["rake1"]))
    if has_tensor(row):
        mt = [float(row["Mrr"]), float(row["Mtt"]), float(row["Mpp"]),
              float(row["Mrt"]), float(row["Mrp"]), float(row["Mtp"])]
        try:
            p1, p2 = mt2plane(mt)  # each is (strike, dip, rake)
            return p1 if int(PREFERRED_DC_PLANE) == 1 else p2
        except Exception:
            return None
    return None

def get_class(row) -> str:
    for k in row.keys():
        if str(k).strip().lower() == "class":
            return str(row[k]).strip()
    return "unclassified"

def class_color(label: str) -> str:
    return CLASS_COLORS.get(label, DEFAULT_COLOR)

def draw_one_png(out_path: str, facecolor: str, mt=None, sdr=None, width_pt=220):
    fig = plt.figure(figsize=(width_pt/72.0, width_pt/72.0), dpi=72)
    fig.patch.set_alpha(0.0)  # transparent outside

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
    fig.savefig(out_path, format=FMT, dpi=DPI_OUT,
                bbox_inches="tight", pad_inches=0.0, transparent=True)
    plt.close(fig)

# ---------- worker (top-level for multiprocessing) ----------
def _render_worker(row_dict):
    """
    row_dict must contain a key '__outdir' with the destination folder.
    Returns ('mt'|'sdr'|'dc'|'skip'|'err', event_id, msg_if_err_or_reason)
    """
    try:
        out_dir = row_dict.get("__outdir")
        if not out_dir:
            return ("err", "", "missing __outdir")

        eid_raw = row_dict.get("id")
        eid = "" if eid_raw is None else str(eid_raw).strip()
        if not eid or eid.lower() == "nan":
            return ("skip", eid, "no id")

        label = get_class(row_dict)
        color = class_color(label)
        out_path = os.path.join(out_dir, f"{eid}.{FMT}")

        if ONLY_DOUBLE_COUPLE:
            sdr = get_sdr_for_dc(row_dict)
            if sdr is None:
                return ("skip", eid, "no DC info")
            draw_one_png(out_path, color, mt=None, sdr=sdr, width_pt=BB_SIZE_PT)
            return ("dc", eid, "")

        # Default mode: prefer full tensor; fall back to SDR
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
        return ("err", str(row_dict.get("id", "")), str(e))

# ---------- catalog helpers ----------
def _is_per_class_file(path: str) -> bool:
    """True if filename equals one of the class CSVs (we skip those)."""
    name = os.path.splitext(os.path.basename(path))[0].strip().lower()
    return name in {c.lower() for c in CLASSES}

def _pick_catalog_csv(src_dir: str) -> str | None:
    """
    Prefer a single '*_classified.csv' (not per-class).
    Fallback: any CSV in the folder that contains a 'class' column.
    """
    # 1) prefer "*_classified.csv"
    cands = [p for p in glob.glob(os.path.join(src_dir, "*.csv"))
             if p.lower().endswith("_classified.csv") and not _is_per_class_file(p)]
    if cands:
        return sorted(cands)[0]

    # 2) fallback: first CSV with a 'class' column
    for p in sorted(glob.glob(os.path.join(src_dir, "*.csv"))):
        if _is_per_class_file(p):
            continue
        try:
            head = pd.read_csv(p, nrows=1)
            if any(str(c).strip().lower() == "class" for c in head.columns):
                return p
        except Exception:
            continue
    return None

def render_catalog_for_source(source_name: str):
    """
    Render one source folder: classified_catalogs/<source_name>
    """
    src_dir = os.path.join(ROOT, source_name)
    if not os.path.isdir(src_dir):
        print(f"[skip] Source folder not found: {src_dir}")
        return

    csv_path = _pick_catalog_csv(src_dir)
    if not csv_path:
        print(f"[skip] No suitable classified CSV found in: {src_dir}")
        return

    out_dir = os.path.join(src_dir, "beachballs")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "id" not in df.columns:
        print(f"[skip] {csv_path} has no 'id' column.")
        return

    rows = df.to_dict("records")
    for r in rows:
        r["__outdir"] = out_dir

    ok_mt = ok_sdr = ok_dc = skipped = errs = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_render_worker, r) for r in rows]
        for fut in as_completed(futures):
            kind, eid, msg = fut.result()
            if kind == "mt":
                ok_mt += 1
            elif kind == "sdr":
                ok_sdr += 1
            elif kind == "dc":
                ok_dc += 1
            elif kind == "skip":
                skipped += 1
            else:
                errs += 1

    total = len(rows)
    mode = "DC-only" if ONLY_DOUBLE_COUPLE else "auto(MT→SDR)"
    print(f"[{source_name}] mode={mode}  MT: {ok_mt}, SDR: {ok_sdr}, DC: {ok_dc}, "
          f"skipped: {skipped}, errors: {errs}, total: {total}.  -> {out_dir}")

# ---------- main ----------
def main():
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Edit this list to choose which folders to render:
    SOURCES_TO_RENDER = [
        "merged",
        # "gcmt",
        # "anss",
        # "isc",
        # "gmt",
        # "full",
    ]
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    if not SOURCES_TO_RENDER:
        print("Nothing to render. Edit SOURCES_TO_RENDER in main().")
        return

    for src in SOURCES_TO_RENDER:
        render_catalog_for_source(src)

if __name__ == "__main__":
    main()
