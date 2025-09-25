from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from cat_handler import paths

FIELDS: Sequence[str] = (
    "id","time_iso","longitude","latitude","depth","mag","mag_type",
    "strike1","dip1","rake1","strike2","dip2","rake2",
    "T_plunge","T_azimuth","N_plunge","N_azimuth","P_plunge","P_azimuth",
    "Mrr","Mtt","Mpp","Mrt","Mrp","Mtp","source","dups","class","sub_depth"
)

NUMERIC_COLS = {
    "longitude","latitude","depth","mag",
    "strike1","dip1","rake1","strike2","dip2","rake2",
    "T_plunge","T_azimuth","N_plunge","N_azimuth","P_plunge","P_azimuth",
    "Mrr","Mtt","Mpp","Mrt","Mrp","Mtp","sub_depth",
}

DEFAULT_INCLUDE_CLASSES = (
    "forearc",
    "crustal_intraarc_shallow",
    "crustal_intraarc_deep",
    "backarc",
)

# -------- manual rows to append (from Stanton-Yonge, et al., 2016 selection --------
MANUAL_ROWS_CSV = """\
STB032381A,1981-03-23T19:28:15,-71.81,-33.96,16.6,6.2,mw,335.0,5.0,73.0,173.0,85.0,91.0,50.0,84.0,1.0,352.0,40.0,261.0,1.027e+18,-5.4000000000000006e+17,-4.87e+17,5.85e+17,-4.676e+18,1.14e+17,gcmt,,slab_interface,33.4850769043
STchoy19850303224726,1985-03-03T22:47:26,-71.62,-33.12,33,8,mw,360,35,105,360,35,105,38.0951572542997,40.0634285773075,20.7156360639714,147.309716862859,44.6690416493965,259.262135157158,7.48E+020,4.89E+019,-7.969E+020,1.912E+020,-6.55E+020,2.9E+018,us,gcmt:M030385A;neic:529084;gem:529084,slab_interface,34.8395805359
STB030485A,1985-03-04T00:32:21,-71.76,-33.23,33,7.3,mwc,234,32,155,346,77,60,49,223,29,353,26,99,5.84E+019,-5.5E+018,-5.289E+019,-7.36E+019,9.282E+019,-5.169E+019,us,gem:529093,intra_slab,29.9401741028
STC031785A,1985-03-17T10:41:45,-71.73,-33.22,43.5,6.6,mw,356.0,26.0,91.0,175.0,64.0,90.0,71.0,84.0,0.0,175.0,19.0,265.0,6.243e+18,-1.93e+17,-6.05e+18,4.93e+17,-4.835e+18,5.09e+17,gcmt,,intra_slab,29.7695789337
STC031985A,1985-03-19T04:01:13,-71.94,-33.63,35.6,6.7,mw,355.0,24.0,90.0,175.0,66.0,90.0,69.0,85.0,0.0,355.0,21.0,265.0,8.667e+18,-2.99e+17,-8.368e+18,7.19e+17,-7.813e+18,7.49e+17,gcmt,,intra_slab,25.8404064178
STC032585A,1985-03-25T05:14:39,-72.52,-34.6,23.9,6.5,mw,10.0,20.0,98.0,181.0,70.0,87.0,65.0,86.0,3.0,182.0,25.0,274.0,2.595e+18,-7.9e+16,-2.516e+18,8999999999999998.0,-3.096e+18,-2.45e+17,gcmt,,slab_interface,21.3393611908
STB081285A,1985-08-12T00:04:53,-73.94,-38.45,17.2,6.0,mw,16.0,15.0,111.0,174.0,76.0,84.0,59.0,76.0,5.0,175.0,31.0,268.0,1.11e+18,-5.4e+16,-1.056e+18,2.7800000000000003e+17,-1.977e+18,-1.05e+17,gcmt,,slab_interface,20.7155399323
STC040304E,2004-04-03T09:57:14,-72.35,-30.04,28.0,5.1,mw,320.0,33.0,50.0,185.0,65.0,113.0,62.0,131.0,21.0,355.0,17.0,258.0,1.47e+17,3.18e+16,-1.79e+17,-3.48e+16,-1.3e+17,6.66e+16,gcmt,,intra_slab,10.377614975
STofficial20100227063411530_30,2010-02-27T06:34:11,-72.898,-36.122,30,8.8,mww,178,77,86,17,14,108,58,82,4,179,32,272,1.04E+022,-3.9E+020,-1E+022,3.04E+021,-1.52E+022,-1.19E+021,us,gcmt:C201002270634A;gcmt:C201002270634A;neic:14340585;gem:14340585,slab_interface,33.1989784241
STchoy20110211200530,2011-02-11T20:05:30,-73.13,-36.47,26,6.9,mww,180,77,86,17,14,107,44.7647985214079,264.686052428259,4.33445012612489,358.997348813813,44.9079207304554,93.3302794603725,9.99E+018,-3.3E+017,-9.66E+018,2.5E+017,-2.09E+019,-1.32E+018,us,gcmt:C201102112005A;gcmt:C201102112005A;neic:602064448;gem:602064448,slab_interface,29.9707660675
STchoy20110214034009,2011-02-14T03:40:09,-72.83,-35.38,21,6.7,mww,189,73,85,26,18,106,44.5506806756492,271.533806758164,5.76001173503429,7.23275083387705,44.8721878968511,102.99623301732,5.6E+018,-2.3E+017,-5.37E+018,-1.27E+018,-9.05E+018,-1.73E+018,us,gcmt:C201102140340A;gcmt:C201102140340A;neic:602035966;gem:602035966,slab_interface,24.217218399
ST201408232232A,2014-08-23T22:32:00,-71.74,-32.76,42,6.4,mw,5,26,92,183,64,89,66,96,1,3,24,272,3.729E+018,5.8E+016,-3.787E+018,-3.07E+017,-4.096E+018,-9.5E+016,gmt,gcmt:C201408232232A;gcmt:C201408232232A;us:b000s5rc;neic:610572067;gem:610572067,intra_slab,28.9251651764
ST201506200210A,2015-06-20T02:10:00,-74.1,-36.35,12,6.4,mw,9,18,84,196,72,92,55,88,7,188,34,282,1.963E+018,-3.3E+017,-1.634E+018,-4.35E+017,-5.009E+018,-7.87E+017,gmt,gcmt:C201506200210A;gcmt:C201506200210A;us:10002ke8;neic:607304048;gem:607304048,slab_interface,11.1978616714
201612251422A,2016-12-25T14:22:00,-74.43,-43.41,33,6.8,mw,4,19,96,178,71,88,61,96,2,3,29,272,1.803E+020,-8.9E+018,-1.713E+020,-2.13E+019,-2.881E+020,1.8E+018,gmt,gcmt:C201612251422A;gcmt:C201612251422A;us:10007mn3;neic:609939179;gem:609939179,intra_slab,22.2438755035
"""

def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has all FIELDS; create missing ones and reorder."""
    for c in FIELDS:
        if c not in df.columns:
            df[c] = np.nan
    return df.loc[:, FIELDS]

def _filter_by_classes(df: pd.DataFrame, include_classes: Iterable[str]) -> pd.DataFrame:
    """Return only rows whose 'class' is in include_classes (case-insensitive)."""
    inc = {str(c).strip().lower() for c in include_classes}
    if "class" not in df.columns:
        raise ValueError("Input catalog must have a 'class' column to filter by classes.")
    mask = df["class"].astype(str).str.strip().str.lower().isin(inc)
    return df.loc[mask].copy()

def _parse_manual_rows(csv_text: str) -> pd.DataFrame:
    """Parse the manual CSV rows into the unified schema with numeric coercion."""
    df = pd.read_csv(io.StringIO(csv_text), header=None, names=FIELDS)
    # Coerce numeric columns; leave strings (id, time_iso, mag_type, source, dups, class)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return _ensure_schema(df)

def build_filtered_plus_manual(
    merged_catalog_csv: str | Path,
    out_csv: str | Path,
    include_classes: Iterable[str] = DEFAULT_INCLUDE_CLASSES,
    manual_rows_csv: str = MANUAL_ROWS_CSV,
    drop_duplicate_ids: bool = True,
) -> None:
    """Read merged classified catalog, filter by classes, append manual rows, and write to CSV."""
    merged_catalog_csv = Path(merged_catalog_csv)
    out_csv = Path(out_csv)
    base = pd.read_csv(merged_catalog_csv)
    base = _ensure_schema(base)
    filtered = _filter_by_classes(base, include_classes)
    manual = _parse_manual_rows(manual_rows_csv)
    combined = pd.concat([filtered, manual], ignore_index=True)
    if drop_duplicate_ids:
        combined = combined.drop_duplicates(subset=["id"], keep="first")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)

if __name__ == "__main__":
    build_filtered_plus_manual(
        merged_catalog_csv=paths.merged_classified_folder,  # classified merged catalog
        out_csv=paths.CLASSIFIED_CATALOGS / "selection_plus_manual.csv",
        include_classes=("forearc", "crustal_intraarc_shallow", "crustal_intraarc_deep", "backarc"),
    )
    pass
