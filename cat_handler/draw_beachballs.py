import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from obspy.imaging.beachball import beachball as bb
from obspy.imaging.beachball import mt2plane

from cat_handler import paths

BB_SIZE_PT  = 220
DPI_OUT     = 50
FMT         = "png"
MAX_WORKERS = None

ONLY_DOUBLE_COUPLE = True
PREFERRED_DC_PLANE = 1  # 1 or 2

# ---------------- Class labels & colors ----------------
CLASS_COLORS: Dict[str, str] = {
    "intraarc_shallow": "darkgreen",
    "intraarc_deep":    "olive",
    "slab_interface":           "deepskyblue",
    "intra_slab":               "teal", #"deepskyblue",
    "outer_rise":               "burlywood",
    "forearc":                  "orange",
    "backarc":                  "mediumpurple",
    "slab_deep":                "firebrick",
    "unclassified":             "#7f7f7f",
}
DEFAULT_COLOR = "#7f7f7f"

CATALOGS: Dict[str, Path] = {
    # "anss":        paths.anss_classified_folder,
    # "gcmt_perez":  paths.gcmt_perez_classified_folder,
    # "gcmt":        paths.gcmt_classified_folder,
    # "isc":         paths.isc_classified_folder,
    # "isc_gem1":     "./isc_gem_chile/isc_gem_chile.csv"
    "selected": paths.selected_classified,
    "relocated": paths.relocated_classified,
    "merged":      paths.merged_classified,
    # "full":        paths.full_classified_folder,
}
OUT_ROOT: Path = paths.BEACHBALLS

def _finite(x: object) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def has_tensor(row: dict) -> bool:
    comps: List[float] = []
    for k in ("Mrr", "Mtt", "Mpp", "Mrt", "Mrp", "Mtp"):
        try:
            f = float(row.get(k))
        except (TypeError, ValueError):
            return False
        if not np.isfinite(f):
            return False
        comps.append(f)
    return any(abs(f) > 1e-12 for f in comps)

def has_sdr(row: dict) -> bool:
    return all(_finite(row.get(k)) for k in ("strike1", "dip1", "rake1"))

def get_sdr_for_dc(row: dict) -> Optional[Tuple[float, float, float]]:
    if has_sdr(row):
        return (float(row["strike1"]), float(row["dip1"]), float(row["rake1"]))
    if has_tensor(row):
        mt = [float(row["Mrr"]), float(row["Mtt"]), float(row["Mpp"]),
              float(row["Mrt"]), float(row["Mrp"]), float(row["Mtp"])]
        try:
            p1, p2 = mt2plane(mt)
            return p1 if int(PREFERRED_DC_PLANE) == 1 else p2
        except Exception:
            return None
    return None

def get_class(row: dict) -> str:
    for k in row.keys():
        if str(k).strip().lower() == "class":
            return str(row[k]).strip()
    return "unclassified"

def class_color(label: str) -> str:
    return CLASS_COLORS.get(label, DEFAULT_COLOR)

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")

def _safe_id(s: object) -> str:
    if s is None:
        return ""
    sid = str(s).strip()
    sid = sid.replace(" ", "_")
    sid = _SANITIZE_RE.sub("_", sid)
    return sid or ""

def draw_one_png(out_path: Path, facecolor: str,
                 mt: Optional[List[float]] = None,
                 sdr: Optional[Tuple[float, float, float]] = None,
                 width_pt: int = BB_SIZE_PT) -> None:
    fig = plt.figure(figsize=(width_pt/72.0, width_pt/72.0), dpi=72)
    fig.patch.set_alpha(0.0)
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), format=FMT, dpi=DPI_OUT,
                bbox_inches="tight", pad_inches=0.0, transparent=True)
    plt.close(fig)

def _render_worker(row_dict: dict) -> Tuple[str, str, str]:
    try:
        out_dir = row_dict.get("__outdir")
        if not out_dir:
            return ("err", "", "missing __outdir")
        eid = _safe_id(row_dict.get("id"))
        if not eid or eid.lower() == "nan":
            return ("skip", eid, "no id")
        label = get_class(row_dict)
        color = class_color(label)
        out_path = Path(out_dir) / f"{eid}.{FMT}"

        if ONLY_DOUBLE_COUPLE:
            sdr = get_sdr_for_dc(row_dict)
            if sdr is None:
                return ("skip", eid, "no DC info")
            draw_one_png(out_path, color, mt=None, sdr=sdr)
            return ("dc", eid, "")

        if has_tensor(row_dict):
            mt = [float(row_dict["Mrr"]), float(row_dict["Mtt"]), float(row_dict["Mpp"]),
                  float(row_dict["Mrt"]), float(row_dict["Mrp"]), float(row_dict["Mtp"])]
            draw_one_png(out_path, color, mt=mt, sdr=None)
            return ("mt", eid, "")
        if has_sdr(row_dict):
            sdr = (float(row_dict["strike1"]), float(row_dict["dip1"]), float(row_dict["rake1"]))
            draw_one_png(out_path, color, mt=None, sdr=sdr)
            return ("sdr", eid, "")
        return ("skip", eid, "no MT/SDR")
    except Exception as e:
        return ("err", _safe_id(row_dict.get("id")), str(e))

def render_catalog(name: str, csv_path: Path, out_root: Path) -> None:
    if not csv_path.exists():
        print(f"[skip] not found: {csv_path}")
        return
    out_dir = out_root / name
    df = pd.read_csv(csv_path)
    if "id" not in df.columns:
        print(f"[skip] {csv_path} has no 'id' column.")
        return

    rows = df.to_dict("records")
    for r in rows:
        r["__outdir"] = str(out_dir)

    ok_mt = ok_sdr = ok_dc = skipped = errs = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_render_worker, r) for r in rows]
        for fut in as_completed(futures):
            kind, _, _ = fut.result()
            if kind == "mt":   ok_mt += 1
            elif kind == "sdr": ok_sdr += 1
            elif kind == "dc":  ok_dc += 1
            elif kind == "skip": skipped += 1
            else: errs += 1

    mode = "DC-only" if ONLY_DOUBLE_COUPLE else "auto(MT→SDR)"
    total = len(rows)
    print(f"[{name}] mode={mode}  MT:{ok_mt} SDR:{ok_sdr} DC:{ok_dc} "
          f"skipped:{skipped} errors:{errs} total:{total} -> {out_dir}")

def main() -> None:
    sources_to_render = [
        # "merged",
        "selected",
        "relocated",
        # "anss",
        # "gcmt_perez",
        # "gcmt",
        # "isc",
        # "isc_gem",
        # "full",
    ]
    if not sources_to_render:
        print("Nothing to render. Edit sources_to_render in main()."); return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name in sources_to_render:
        csv_path = CATALOGS.get(name)
        if not csv_path:
            print(f"[skip] unknown source key: {name}")
            continue
        render_catalog(name, Path(csv_path), OUT_ROOT)

if __name__ == "__main__":
    main()
