
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from obspy.imaging.beachball import beachball as bb

# ---------- CONFIG ----------
INPUT_CSV   = "catalog_labeled.csv"
OUT_DIR     = "beachballs_svg"
BB_SIZE_PT  = 220

# Your only valid classes:
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

# Fixed colors (tweak if you like)
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
DEFAULT_COLOR = "#7f7f7f"  # safety

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
    # no normalization beyond trimming – you said labels are exact
    return CLASS_COLORS.get(label, DEFAULT_COLOR)

def draw_one_svg(out_path: str, facecolor: str, mt=None, sdr=None, width_pt=220):
    # Single, isolated figure per event; always overwrite
    fig = plt.figure(figsize=(width_pt/72.0, width_pt/72.0), dpi=72)
    fig.patch.set_alpha(0.0)  # transparent figure background

    # Draw beachball. Note: beachball 'bgcolor' is the ball background;
    # we keep it 'w' and rely on transparent figure outside the ball.
    if mt is not None:
        bb(mt, width=width_pt, facecolor=facecolor, edgecolor="black",
           linewidth=0.8, bgcolor="w", fig=fig)
    else:
        bb(tuple(sdr), width=width_pt, facecolor=facecolor, edgecolor="black",
           linewidth=0.8, bgcolor="w", fig=fig)

    # Make axes transparent & clean
    for ax in fig.axes:
        ax.set_facecolor("none")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, format="png", dpi=30, bbox_inches="tight", pad_inches=0.0, transparent=True)
    plt.close(fig)

# ---------- main ----------
def main():
    df = pd.read_csv(INPUT_CSV)
    if "id" not in df.columns:
        raise SystemExit("CSV must have an 'id' column.")

    total = len(df)
    ok_mt = ok_sdr = skipped = 0

    for i, row in df.iterrows():

        eid = str(row.get("id") or "").strip()
        if not eid or eid.lower() == "nan":
            skipped += 1
            continue

        label = get_class(row)
        color = class_color(label)

        out_path = os.path.join(OUT_DIR, f"{eid}.png")

        if has_tensor(row):
            mt = [float(row["Mrr"]), float(row["Mtt"]), float(row["Mpp"]),
                  float(row["Mrt"]), float(row["Mrp"]), float(row["Mtp"])]
            try:
                draw_one_svg(out_path, color, mt=mt, sdr=None, width_pt=BB_SIZE_PT)
                ok_mt += 1
            except Exception as e:
                print(f"[warn] tensor failed for {eid}: {e}")
                skipped += 1

        elif has_sdr(row):
            sdr = (float(row["strike1"]), float(row["dip1"]), float(row["rake1"]))
            try:
                draw_one_svg(out_path, color, mt=None, sdr=sdr, width_pt=BB_SIZE_PT)
                ok_sdr += 1
            except Exception as e:
                print(f"[warn] sdr failed for {eid}: {e}")
                skipped += 1

        else:
            # nothing to draw
            skipped += 1

    print(f"Done. Rendered MT: {ok_mt}, SDR: {ok_sdr}, skipped: {skipped}, total rows: {total}")
    print(f"SVGs written to: {OUT_DIR}")

if __name__ == "__main__":
    main()
