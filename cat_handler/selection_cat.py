# select_events_from_relocated.py
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Sequence, Dict, Any
from cat_handler import paths

def _to_num(s, default=np.nan):
    v = pd.to_numeric(pd.Series([s]), errors="coerce").iloc[0]
    return float(v) if pd.notna(v) else default

def _ensure_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

def build_selected_catalog(
    relocated_classified_csv: str,
    stanton_csv: str,
    out_csv: str,
    include_classes: Sequence[str] = ("forearc", "backarc", "crustal_intraarc_shallow", "crustal_intraarc_deep"),
    match_classes: Sequence[str] = ("slab_interface", "forearc", "intra_slab"),
    mag_tolerance: float = 0.5,
    date_window_days: int = 0,   # 0 = same date only
    st_prefix: str | None = "st_",     # prefix IDs of Stanton-matched events
    manual_rows: Sequence[Dict[str, Any]] | None = None,  # extra rows to append
) -> None:
    # --- load relocated (already classified) ---
    df = pd.read_csv(relocated_classified_csv)

    # base selection by class
    base = df[df["class"].isin(include_classes)].copy()

    # prep matching candidates (restrict classes for matching)
    cand = df[df["class"].isin(match_classes)].copy()
    cand["_date"] = pd.to_datetime(cand["time_iso"], errors="coerce", utc=True).dt.date
    cand["_mag"]  = pd.to_numeric(cand["mag"], errors="coerce")
    cand = cand[cand["_date"].notna()]

    # --- load Stanton ---
    st = pd.read_csv(stanton_csv)
    st["_date"] = pd.to_datetime(st["Date"], dayfirst=True, errors="coerce").dt.date
    st["_mag"]  = pd.to_numeric(st["Mw"], errors="coerce")

    matched_idx: set[int] = set()

    if not cand.empty:
        for _, r in st.iterrows():
            sd = r["_date"]
            smag = _to_num(r["_mag"])
            if pd.isna(sd):
                continue

            # date window
            low = sd - timedelta(days=date_window_days)
            high = sd + timedelta(days=date_window_days)
            sub = cand[(cand["_date"] >= low) & (cand["_date"] <= high)]
            if sub.empty:
                continue

            # magnitude gate (if Stanton mag available)
            if np.isfinite(smag):
                sub = sub.assign(_dmag=(sub["_mag"] - smag).abs())
                sub = sub[sub["_dmag"] <= mag_tolerance]
                if sub.empty:
                    continue
                i_best = int(sub["_dmag"].idxmin())
            else:
                # if Stanton Mw missing, take the first candidate on that date
                i_best = int(sub.index[0])

            matched_idx.add(i_best)

    # Stanton-matched additions
    extra = df.loc[sorted(matched_idx)].copy()
    if st_prefix:
        extra["id"] = st_prefix + extra["id"].astype(str)

    # Manual additions (if any)
    if manual_rows:
        manual_df = pd.DataFrame(manual_rows)
        # make sure manual rows have the same columns as df
        manual_df = _ensure_columns(manual_df, df.columns)
    else:
        manual_df = pd.DataFrame(columns=df.columns)

    # Final assembly: base + Stanton matches + manual rows
    final = pd.concat([base, extra, manual_df[df.columns]], ignore_index=True)

    # Deduplicate by id; keep='last' so manual rows override if duplicated
    final = final.drop_duplicates(subset=["id"], keep="last")

    final.to_csv(out_csv, index=False)
    print(f"[OK] {out_csv}  base={len(base)} matched_add={len(extra)} manual_add={len(manual_df)} total={len(final)}")

def main():
    # Your manual row:
    chiloe_eq = [{
        "id": "10007mn3",
        "time_iso": "2016-12-25T14:22:42",
        "longitude": -74.229771,
        "latitude": -43.534537,
        "depth": 35.3278,
        "mag": 7.6,
        "mag_type": "mww",
        "lon_error": "relocated",
        "lat_error": "relocated",
        "depth_error": "relocated",
        "mag_error": np.nan,
        "strike1": 183.0, "dip1": 74.0, "rake1": 92.0,
        "strike2": 356.0, "dip2": 16.0, "rake2": 83.0,
        "Mrr": 1.803e20, "Mtt": -8.9e18, "Mpp": -1.713e20,
        "Mrt": -2.13e19, "Mrp": -2.881e20, "Mtp": 1.8e18,
        "source": "anss",
        "dups": "gcmt:C201612251422A",
        "class": "intra_slab",
        "sub_depth": 28.1089038849,
    }]

    build_selected_catalog(
        relocated_classified_csv=str(paths.relocated_classified),
        stanton_csv=str(paths.rawcat_stanton),
        out_csv=str(paths.selected_classified),
        include_classes=("forearc", "backarc", "crustal_intraarc_shallow", "crustal_intraarc_deep"),
        match_classes=("slab_interface", "forearc", "intra_slab"),
        mag_tolerance=0.3,
        date_window_days=0,
        st_prefix="",          # set to None to avoid prefixing Stanton matches
        manual_rows=chiloe_eq,       # <— your manual additions here
    )

if __name__ == "__main__":
    main()
