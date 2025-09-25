#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd

# Must be set before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from obspy.imaging.beachball import beachball as bb

# ---------- CONFIG ----------
INPUT_CSV  = "./locals/catalog_local_fm.csv"
OUT_DIR    = "locals/local_fm"        # output folder
BB_SIZE_PT = 220               # beachball width in points

# Colors per class (if a 'class' column exists); else falls back to DEFAULT_COLOR
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

def class_color(label: str) -> str:
    if not isinstance(label, str):
        return DEFAULT_COLOR
    return CLASS_COLORS.get(label.strip(), DEFAULT_COLOR)

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    # lower-case, strip, remove spaces/underscores for robust lookup
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _get_first_existing(row: dict, names: list):
    for n in names:
        if n in row and row[n] not in (None, "", "nan", "NaN"):
            return row[n]
    return None

_SPLIT_RE = re.compile(r"[;/,\s]+")

def _parse_plane_value(val) -> tuple | None:
    """
    Parse a 'p1'/'p2' cell that may look like:
      'strike,dip,rake' or 'strike dip rake' (any mix of separators).
    Returns (strike, dip, rake) as floats or None.
    """
    if val is None:
        return None
    if not isinstance(val, str):
        # maybe already a tuple-like or a 3-list?
        try:
            seq = list(val)
            if len(seq) == 3 and all(_finite(x) for x in seq):
                return float(seq[0]), float(seq[1]), float(seq[2])
        except Exception:
            return None
        return None

    parts = [p for p in _SPLIT_RE.split(val.strip()) if p != ""]
    if len(parts) < 3:
        return None
    try:
        s, d, r = float(parts[0]), float(parts[1]), float(parts[2])
        return s, d, r
    except Exception:
        return None

def _extract_sdr(row: dict) -> tuple | None:
    """
    Try to extract SDR in this order:
      1) composite 'p1' column (e.g., 'strike,dip,rake')
      2) split p1_* columns: p1_strike/p1_dip/p1_rake
      3) generic split: strike1/dip1/rake1
      4) fallback to 'p2' / p2_* / strike2,dip2,rake2
    Returns (strike, dip, rake) or None.
    """
    # 1) composite p1
    for key in ("p1", "P1"):
        if key in row:
            sdr = _parse_plane_value(row[key])
            if sdr and all(_finite(v) for v in sdr):
                return sdr

    # 2) split p1_*
    p1s = _get_first_existing(row, ["p1_Strike", "p1strike", "strike_p1"])
    p1d = _get_first_existing(row, ["p1_Dip", "p1dip", "dip_p1"])
    p1r = _get_first_existing(row, ["p1_Rake", "p1rake", "rake_p1"])
    if all(_finite(v) for v in (p1s, p1d, p1r)):
        return float(p1s), float(p1d), float(p1r)

    # 3) generic strike1/dip1/rake1
    g1s = _get_first_existing(row, ["strike1", "s1", "strike_1"])
    g1d = _get_first_existing(row, ["dip1", "d1", "dip_1"])
    g1r = _get_first_existing(row, ["rake1", "r1", "rake_1"])
    if all(_finite(v) for v in (g1s, g1d, g1r)):
        return float(g1s), float(g1d), float(g1r)

    # 4) fallback to p2
    for key in ("p2", "P2"):
        if key in row:
            sdr = _parse_plane_value(row[key])
            if sdr and all(_finite(v) for v in sdr):
                return sdr

    p2s = _get_first_existing(row, ["p2_Strike", "p2strike", "strike_p2"])
    p2d = _get_first_existing(row, ["p2_Dip", "p2dip", "dip_p2"])
    p2r = _get_first_existing(row, ["p2_Rake", "p2rake", "rake_p2"])
    if all(_finite(v) for v in (p2s, p2d, p2r)):
        return float(p2s), float(p2d), float(p2r)

    g2s = _get_first_existing(row, ["strike2", "s2", "strike_2"])
    g2d = _get_first_existing(row, ["dip2", "d2", "dip_2"])
    g2r = _get_first_existing(row, ["rake2", "r2", "rake_2"])
    if all(_finite(v) for v in (g2s, g2d, g2r)):
        return float(g2s), float(g2d), float(g2r)

    return None

def _get_id(row: dict, index: int) -> str:
    for key in ("id", "ID", "eventid", "event_id", "name"):
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
    return f"row_{index:06d}"

def _get_label(row: dict) -> str:
    # optional; if absent, use 'unclassified'
    for key in row.keys():
        if str(key).strip().lower() == "class":
            return str(row[key]).strip()
    return "unclassified"

def draw_one_png(out_path: str, facecolor: str, sdr, width_pt=BB_SIZE_PT):
    fig = plt.figure(figsize=(width_pt/72.0, width_pt/72.0), dpi=72)
    fig.patch.set_alpha(0.0)
    bb(tuple(sdr), width=width_pt, facecolor=facecolor, edgecolor="black",
       linewidth=0.8, bgcolor="w", fig=fig)
    for ax in fig.axes:
        ax.set_facecolor("none")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # If you prefer SVGs, change format='png'->'svg' and filename extension accordingly.
    fig.savefig(out_path, format="png", dpi=30, bbox_inches="tight",
                pad_inches=0.0, transparent=True)
    plt.close(fig)

# ---------- main ----------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    df = _norm_cols(df)
    rows = df.to_dict("records")

    ok, skip = 0, 0
    for i, row in enumerate(rows):
        eid = _get_id(row, i)
        sdr = _extract_sdr(row)
        print(sdr)
        if not sdr:
            skip += 1
            continue
        label = _get_label(row)
        color = class_color(label)
        out_path = os.path.join(OUT_DIR, f"{eid}.png")  # change to .svg if you prefer
        try:
            draw_one_png(out_path, color, sdr)
            ok += 1
        except Exception as e:
            print(f"[warn] failed {eid}: {e}")
            skip += 1

    print(f"Done. Rendered: {ok}, skipped: {skip}, total rows: {len(rows)}")
    print(f"Files written in: {OUT_DIR}")

if __name__ == "__main__":
    main()
