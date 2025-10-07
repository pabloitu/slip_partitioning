# build_selection_plus_manual_and_fix_mechs.py
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Sequence, List, Tuple

import numpy as np
import pandas as pd

from cat_handler import paths

# ---------------------------- unified schema (NO T/N/P axes) ----------------------------
FIELDS: Sequence[str] = (
        "id", "time_iso", "longitude", "latitude", "depth", "mag", "mag_type",
        "lon_error", "lat_error", "depth_error", "mag_error",
        "strike1", "dip1", "rake1", "strike2", "dip2", "rake2",
        "Mrr", "Mtt", "Mpp", "Mrt", "Mrp", "Mtp",
        "source", "dups", "class", "sub_depth"
)
NUMERIC_COLS = {
    "longitude","latitude","depth","mag","lon_error", "lat_error", "depth_error", "mag_error",
    "strike1","dip1","rake1","strike2","dip2","rake2",
    "Mrr","Mtt","Mpp","Mrt","Mrp","Mtp","sub_depth",
}
SDR_FIELDS = ("strike1","dip1","rake1","strike2","dip2","rake2")
MT_FIELDS  = ("Mrr","Mtt","Mpp","Mrt","Mrp","Mtp")

DEFAULT_INCLUDE_CLASSES = (
    "forearc",
    "intraarc_shallow",
    "intraarc_deep",
    "backarc",
)

# -------- manual rows to append (T/N/P removed) --------
# MANUAL_ROWS_CSV = """\
# STB032381A,1981-03-23T19:28:15,-71.81,-33.96,16.6,6.2,mw,335.0,5.0,73.0,173.0,85.0,91.0,1.027e+18,-5.4000000000000006e+17,-4.87e+17,5.85e+17,-4.676e+18,1.14e+17,gcmt,,slab_interface,33.4850769043
# STchoy19850303224726,1985-03-03T22:47:26,-71.62,-33.12,40.0,8.0,mw,,,,,360.0,35.0,105.0,360.0,35.0,105.0,7.48e+20,4.89e+19,-7.969e+20,1.912e+20,-6.55e+20,2.9e+18,anss,gcmt:M030385A,slab_interface,34.8395805359
# STB030485A,1985-03-04T00:32:21,-71.76,-33.23,33,7.3,mwc,234,32,155,346,77,60,5.84E+019,-5.5E+018,-5.289E+019,-7.36E+019,9.282E+019,-5.169E+019,us,gem:529093,intra_slab,29.9401741028
# STC031785A,1985-03-17T10:41:45,-71.73,-33.22,43.5,6.6,mw,356.0,26.0,91.0,175.0,64.0,90.0,6.243e+18,-1.93e+17,-6.05e+18,4.93e+17,-4.835e+18,5.09e+17,gcmt,,intra_slab,29.7695789337
# STC031985A,1985-03-19T04:01:13,-71.94,-33.63,35.6,6.7,mw,355.0,24.0,90.0,175.0,66.0,90.0,8.667e+18,-2.99e+17,-8.368e+18,7.19e+17,-7.813e+18,7.49e+17,gcmt,,intra_slab,25.8404064178
# STC032585A,1985-03-25T05:14:39,-72.52,-34.6,23.9,6.5,mw,10.0,20.0,98.0,181.0,70.0,87.0,2.595e+18,-7.9e+16,-2.516e+18,8999999999999998.0,-3.096e+18,-2.45e+17,gcmt,,slab_interface,21.3393611908
# STB081285A,1985-08-12T00:04:53,-73.94,-38.45,17.2,6.0,mw,16.0,15.0,111.0,174.0,76.0,84.0,1.11e+18,-5.4e+16,-1.056e+18,2.7800000000000003e+17,-1.977e+18,-1.05e+17,gcmt,,slab_interface,20.7155399323
# STC040304E,2004-04-03T09:57:14,-72.35,-30.04,28.0,5.1,mw,320.0,33.0,50.0,185.0,65.0,113.0,1.47e+17,3.18e+16,-1.79e+17,-3.48e+16,-1.3e+17,6.66e+16,gcmt,,intra_slab,10.377614975
# STofficial20100227063411530_30,2010-02-27T06:34:11,-72.898,-36.122,30,8.8,mww,178,77,86,17,14,108,1.04E+022,-3.9E+020,-1E+022,3.04E+021,-1.52E+022,-1.19E+021,us,gcmt:C201002270634A;gcmt:C201002270634A;neic:14340585;gem:14340585,slab_interface,33.1989784241
# STchoy20110211200530,2011-02-11T20:05:30,-73.13,-36.47,26,6.9,mww,180,77,86,17,14,107,9.99E+018,-3.3E+017,-9.66E+018,2.5E+017,-2.09E+019,-1.32E+018,us,gcmt:C201102112005A;gcmt:C201102112005A;neic:602064448;gem:602064448,slab_interface,29.9707660675
# STchoy20110214034009,2011-02-14T03:40:09,-72.83,-35.38,21,6.7,mww,189,73,85,26,18,106,5.6E+018,-2.3E+017,-5.37E+018,-1.27E+018,-9.05E+018,-1.73E+018,us,gcmt:C201102140340A;gcmt:C201102140340A;neic:602035966;gem:602035966,slab_interface,24.217218399
# ST201408232232A,2014-08-23T22:32:00,-71.74,-32.76,42,6.4,mw,5,26,92,183,64,89,3.729E+018,5.8E+016,-3.787E+018,-3.07E+017,-4.096E+018,-9.5E+016,gmt,gcmt:C201408232232A;gcmt:C201408232232A;us:b000s5rc;neic:610572067;gem:610572067,intra_slab,28.9251651764
# ST201506200210A,2015-06-20T02:10:00,-74.1,-36.35,12,6.4,mw,9,18,84,196,72,92,1.963E+018,-3.3E+017,-1.634E+018,-4.35E+017,-5.009E+018,-7.87E+017,gmt,gcmt:C201506200210A;gcmt:C201506200210A;us:10002ke8;neic:607304048;gem:607304048,slab_interface,11.1978616714
# 10007mn3,2016-12-25T14:22:42,-74.229771,-43.534537,35.3278,7.6,mww,relocated,relocated,relocated,,183.0,74.0,92.0,356.0,16.0,83.0,1.803e+20,-8.9e+18,-1.713e+20,-2.13e+19,-2.881e+20,1.8e+18,anss,gcmt:C201612251422A,slab_interface,28.1089038849
# """

MANUAL_ROWS_CSV = """\
ST_B032381A,1981-03-23T19:28:10,-71.89,-33.66,46.0,6.4,mwc,,,2.6,,335.0,5.0,73.0,173.0,85.0,91.0,1.027e+18,-5.4000000000000006e+17,-4.87e+17,5.85e+17,-4.676e+18,1.14e+17,anss,,slab_interface,27.3459835052
ST_choy19850303224726,1985-03-03T22:47:26,-71.62,-33.12,40.0,8.0,mw,,,,,360.0,35.0,105.0,360.0,35.0,105.0,7.48e+20,4.89e+19,-7.969e+20,1.912e+20,-6.55e+20,2.9e+18,anss,gcmt:M030385A,slab_interface,34.8395805359
ST_C031785A,1985-03-17T10:41:37,-71.73,-33.22,43.5,6.5,mwc,0.02,0.01,0.8,,356.0,26.0,91.0,175.0,64.0,90.0,6.243e+18,-1.93e+17,-6.05e+18,4.93e+17,-4.835e+18,5.09e+17,anss,,slab_interface,29.7695789337
ST_C031985A,1985-03-19T04:01:06,-71.94,-33.63,35.6,6.6,mwc,0.01,0.01,0.8,,355.0,24.0,90.0,175.0,66.0,90.0,8.667e+18,-2.99e+17,-8.368e+18,7.19e+17,-7.813e+18,7.49e+17,anss,,slab_interface,25.8404064178
ST_C032585A,1985-03-25T05:14:33,-72.52,-34.6,23.9,6.3,mwc,0.02,0.02,1.0,,10.0,20.0,98.0,181.0,70.0,87.0,2.595e+18,-7.9e+16,-2.516e+18,8999999999999998.0,-3.096e+18,-2.45e+17,anss,,slab_interface,21.3393611908
ST_B081285A,1985-08-12T00:04:50,-73.94,-38.45,17.2,6.2,mwc,0.03,0.02,1.3,,16.0,15.0,111.0,174.0,76.0,84.0,1.11e+18,-5.4e+16,-1.056e+18,2.7800000000000003e+17,-1.977e+18,-1.05e+17,anss,,slab_interface,20.7155399323
ST_official20100227063411530_30,2010-02-27T06:34:11,-72.898,-36.122,30.0,8.8,mww,,,default,,178.0,77.0,86.0,17.0,14.0,108.0,1.04e+22,-3.9e+20,-1e+22,3.04e+21,-1.52e+22,-1.19e+21,anss,gcmt:C201002270634A,slab_interface,33.1989784241
ST_choy20110211200530,2011-02-11T20:05:30,-73.662876,-36.741835,22.9166,6.9,mww,relocated,relocated,relocated,,180.0,77.0,86.0,17.0,14.0,107.0,9.99e+18,-3.3e+17,-9.66e+18,2.5e+17,-2.09e+19,-1.32e+18,anss,gcmt:C201102112005A,slab_interface,20.3735370636
ST_choy20110214034009,2011-02-14T03:40:09,-73.673771,-35.515663,4.1811,6.7,mww,relocated,relocated,relocated,,189.0,73.0,85.0,26.0,18.0,106.0,5.600000000000001e+18,-2.3e+17,-5.37e+18,-1.27e+18,-9.05e+18,-1.7299999999999995e+18,anss,gcmt:C201102140340A,slab_interface,11.5729513168
ST_b000s5rc,2014-08-23T22:32:32,-71.531694,-32.730821,36.3984,6.4,mww,relocated,relocated,relocated,,183.0,69.0,92.0,359.0,21.0,86.0,3.729e+18,5.8e+16,-3.787e+18,-3.07e+17,-4.096e+18,-9.5e+16,anss,gcmt:C201408232232A,slab_interface,36.2186012268
ST_10007mn3,2016-12-25T14:22:42,-74.229771,-43.534537,35.3278,7.6,mww,relocated,relocated,relocated,,183.0,74.0,92.0,356.0,16.0,83.0,1.803e+20,-8.9e+18,-1.713e+20,-2.13e+19,-2.881e+20,1.8e+18,anss,gcmt:C201612251422A,slab_interface,28.1089038849
"""

# MARK = 'STC032585A'

# IDs to remove from the final selection
TO_REMOVE: List[str] = [
    "B081496F",
    # "6000nmw5",
    # "7000jw73",
    # "B031681A",
    # "B041403A",
    # "C040985A",
    # "60006c5a",
    # "B032381A",
    # "B030485B",
    # "C031785A",
    # "pde20120325223706000_40",
    # "10002ke8",
    # 'STB030485A'
]

# Mechanism overrides: (target_anss_id, source_gcmt_id)
MECH_OVERRIDES: List[Tuple[str, str]] = [
    ("pde20040503043650040_21", "C050304A"),
]

# ---------------------------- helpers ----------------------------
def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has all FIELDS and reorder."""
    for c in FIELDS:
        if c not in df.columns:
            df[c] = np.nan
    return df.loc[:, FIELDS]

def _filter_by_classes(df: pd.DataFrame, include_classes: Iterable[str]) -> pd.DataFrame:
    """Keep rows whose 'class' is in include_classes (case-insensitive)."""
    inc = {str(c).strip().lower() for c in include_classes}
    if "class" not in df.columns:
        raise ValueError("Input catalog must have a 'class' column to filter by classes.")
    mask = df["class"].astype(str).str.strip().str.lower().isin(inc)
    return df.loc[mask].copy()

def _parse_manual_rows(csv_text: str) -> pd.DataFrame:
    """Parse manual CSV rows into the unified schema with numeric coercion."""
    df = pd.read_csv(io.StringIO(csv_text), header=None, names=FIELDS)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return _ensure_schema(df)

def _drop_excluded_ids(df: pd.DataFrame, exclude_ids: Iterable[str]) -> pd.DataFrame:
    """Drop rows whose id matches any in exclude_ids; also drop if id without leading 'st' matches."""
    banned = {str(x).strip().lower() for x in exclude_ids}
    ids = df["id"].astype(str).str.lower()
    ids_nost = ids.where(~ids.str.startswith("st"), ids.str[2:])  # remove 'st' prefix if present
    keep = ~(ids.isin(banned) | ids_nost.isin(banned))
    return df.loc[keep].copy()

def _norm_id(s: str) -> str:
    """Normalize IDs by dropping 'gcmt:'/'anss:'/'us:' prefixes."""
    s = str(s).strip()
    low = s.lower()
    for pref in ("gcmt:", "anss:", "us:", "usgs:"):
        if low.startswith(pref):
            return s[len(pref):]
    return s

def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def _load_catalog_for_lookup(path: str | Path) -> pd.DataFrame:
    """Load a source catalog and ensure SDR/MT columns exist (numeric)."""
    df = pd.read_csv(path)
    _ensure_schema(df)
    _coerce_numeric(df, SDR_FIELDS + MT_FIELDS)
    df["id"] = df["id"].astype(str)
    return df

def _apply_mech_overrides_from_gcmt(
    df_final: pd.DataFrame,
    df_gcmt: pd.DataFrame,
    overrides: Iterable[Tuple[str, str]],
    copy_tensor: bool = False,
) -> int:
    """
    For each (anss_id, gcmt_id), overwrite df_final[anss_id]'s SDR (and optionally MT)
    from df_gcmt[gcmt_id]. Returns count of successful overrides.
    """
    gcmt_lookup = { _norm_id(rid): i for i, rid in enumerate(df_gcmt["id"].astype(str)) }

    n_ok = 0
    for anss_id_raw, gcmt_id_raw in overrides:
        anss_id = _norm_id(anss_id_raw)
        gcmt_id = _norm_id(gcmt_id_raw)

        # find target in final
        idx_target = df_final.index[df_final["id"].astype(str) == anss_id]
        if len(idx_target) == 0:
            print(f"[override] target ANSS id not found in final: {anss_id_raw}")
            continue

        # find source in gcmt
        j = gcmt_lookup.get(gcmt_id)
        if j is None:
            print(f"[override] source GCMT id not found in GCMT catalog: {gcmt_id_raw}")
            continue

        # copy SDR (and optionally tensor)
        for i in idx_target:
            for k in SDR_FIELDS:
                df_final.at[i, k] = df_gcmt.at[j, k]
            if copy_tensor:
                for k in MT_FIELDS:
                    df_final.at[i, k] = df_gcmt.at[j, k]
        n_ok += len(idx_target)
    return n_ok

# ---------------------------- main builder ----------------------------
def build_filtered_plus_manual_and_fix_mechs(
    merged_catalog_csv: str | Path,
    out_csv: str | Path,
    include_classes: Iterable[str] = DEFAULT_INCLUDE_CLASSES,
    manual_rows_csv: str = MANUAL_ROWS_CSV,
    drop_duplicate_ids: bool = True,
    exclude_ids: Iterable[str] | None = None,
    # mechanism override inputs
    gcmt_catalog_csv: str | Path = paths.cat_gcmt,
    anss_catalog_csv: str | Path = paths.cat_anss,  # not strictly needed but available for future checks
    mech_overrides: Iterable[Tuple[str, str]] = MECH_OVERRIDES,
    copy_tensor_from_gcmt: bool = False,
) -> None:
    """
    Build final selection: filter by class, append manual rows, drop duplicates,
    drop unwanted ids, then overwrite focal mechanisms for specific ANSS ids using GCMT.
    """
    merged_catalog_csv = Path(merged_catalog_csv)
    out_csv = Path(out_csv)

    # 1) base -> filter classes
    base = pd.read_csv(merged_catalog_csv)
    base = _ensure_schema(base)
    filtered = _filter_by_classes(base, include_classes)

    # 2) append manual rows (already matches schema without T/N/P)
    manual = _parse_manual_rows(manual_rows_csv)
    combined = pd.concat([filtered, manual], ignore_index=True)

    # 3) exclusions
    if exclude_ids:
        combined = _drop_excluded_ids(combined, exclude_ids)

    # 4) (optional) drop duplicate ids
    if drop_duplicate_ids:
        combined = combined.drop_duplicates(subset=["id"], keep="first")

    # 5) mechanism overrides from GCMT → target ANSS rows
    df_gcmt = _load_catalog_for_lookup(gcmt_catalog_csv)
    _ = _load_catalog_for_lookup(anss_catalog_csv)  # not used, but kept for symmetry/future checks

    n_changed = _apply_mech_overrides_from_gcmt(
        df_final=combined,
        df_gcmt=df_gcmt,
        overrides=mech_overrides,
        copy_tensor=copy_tensor_from_gcmt,
    )
    if n_changed:
        print(f"[override] Updated mechanisms for {n_changed} row(s) from GCMT.")

    # 6) write
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}  ({len(combined)} rows)")

# ---------------------------- CLI ----------------------------
if __name__ == "__main__":
    build_filtered_plus_manual_and_fix_mechs(
        merged_catalog_csv=paths.relocated_classified,   # classified merged catalog (input)
        out_csv=paths.selected_classified,                   # final selection (output)
        include_classes=("forearc", "intraarc_shallow", "backarc"),
        manual_rows_csv=MANUAL_ROWS_CSV,
        drop_duplicate_ids=True,
        exclude_ids=TO_REMOVE,
        gcmt_catalog_csv=paths.cat_gcmt,
        anss_catalog_csv=paths.cat_anss,
        mech_overrides=MECH_OVERRIDES,
        copy_tensor_from_gcmt=False,  # True if you also want to replace the moment tensor
    )
