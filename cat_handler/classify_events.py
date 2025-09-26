# classify_relocated.py
from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from scipy.spatial import cKDTree

from cat_handler import paths

# ---- Tunables ----
INTRA_ARC_SHALLOW_MAX = 40.0
DEEP_SLAB_TOL = 10.0
SUBDUCTION_CLASSIFY_MAX_SLAB_DEPTH = 70.0
INTERFACE_DEPTH_TOL = 10.0            # buffer for relocated or “no-error-info” cases
STRICT_INTERFACE_DEPTH_TOL = 15.0    # treat explicit 0 depth_error as poorly constrained
BACKARC_MAX_DEPTH = 50.0
SLAB_QUERY_MAXDIST_KM = 15.0
SLAB_DEPTH_IS_POSITIVE_DOWN = False
CONV_AZIMUTH_DEG = 78.0
NORM_CONE_DEG = 35.0
SLIP_CONE_DEG = 70.0

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
SUBDUCTION_CLASSES = {"slab_interface", "intra_slab", "slab_deep", "forearc"}

# ---- Slab grid ----
@dataclass
class SlabGrid:
    tree: cKDTree
    lon: np.ndarray
    lat: np.ndarray
    depth: np.ndarray   # km, +down
    strike: np.ndarray  # deg
    dip: np.ndarray     # deg

def _lon0360_to_180(x: float) -> float:
    x = float(x)
    return x - 360.0 if x > 180.0 else x

def _read_xyz(p: str) -> pd.DataFrame:
    df = pd.read_csv(p, header=None, names=["lon", "lat", "val"])
    df["lon"] = df["lon"].astype(float).apply(_lon0360_to_180)
    df["lat"] = df["lat"].astype(float)
    return df

def load_slab_xyz(depth_path: str, strike_path: str, dip_path: str) -> SlabGrid:
    dep = _read_xyz(depth_path)
    st  = _read_xyz(strike_path)
    di  = _read_xyz(dip_path)
    if not SLAB_DEPTH_IS_POSITIVE_DOWN:
        dep["val"] = -dep["val"]
    df = dep.merge(st, on=["lon","lat"], how="outer", suffixes=("_dep","_st"))
    df = df.merge(di, on=["lon","lat"], how="outer")
    df.rename(columns={"val":"val_dip"}, inplace=True)
    df = df.dropna(subset=["val_dep","val_st","val_dip"], how="any")
    lon = df["lon"].to_numpy(float)
    lat = df["lat"].to_numpy(float)
    depth  = df["val_dep"].to_numpy(float)
    strike = (df["val_st"].to_numpy(float) % 360.0)
    dip    = df["val_dip"].to_numpy(float)
    return SlabGrid(tree=cKDTree(np.c_[lon, lat]), lon=lon, lat=lat, depth=depth, strike=strike, dip=dip)

# ---- Simple geo helpers ----
def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl   = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def query_slab(grid: SlabGrid, lon: float, lat: float, maxdist_km: float
               ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if np.isnan(lon) or np.isnan(lat):
        return (None, None, None)
    _, idx = grid.tree.query(np.array([lon, lat]), k=1)
    if idx is None:
        return (None, None, None)
    if _haversine_km(lon, lat, grid.lon[idx], grid.lat[idx]) > maxdist_km:
        return (None, None, None)
    return float(grid.depth[idx]), float(grid.strike[idx]), float(grid.dip[idx])

def load_union(path: str):
    if not os.path.exists(path):
        return None
    gdf = gpd.read_file(path).to_crs(epsg=4326)
    return unary_union(gdf.geometry)

def is_west_of_trench(lon: float, lat: float, trench_line) -> bool:
    if trench_line is None or np.isnan(lon) or np.isnan(lat):
        return False
    pt = Point(lon, lat)
    nearest = trench_line.interpolate(trench_line.project(pt))
    return float(lon) < float(nearest.x)

def is_east_of_polygon(lon: float, lat: float, polygon) -> bool:
    if polygon is None or np.isnan(lon) or np.isnan(lat):
        return False
    boundary = polygon.boundary
    pt = Point(lon, lat)
    nearest = boundary.interpolate(boundary.project(pt))
    return float(lon) > float(nearest.x)

# ---- Focal-mechanism cones ----
def _deg2rad(x): return math.radians(float(x))
def _clamp(x):   return max(-1.0, min(1.0, float(x)))

def _angle_between(a: np.ndarray, b: np.ndarray, oriented: bool=False) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float('nan')
    a, b = a/na, b/nb
    dot = float(np.dot(a, b))
    if not oriented:
        dot = abs(dot)
    return math.degrees(math.acos(_clamp(dot)))

def _strike_dip_axes(strike_deg: float, dip_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    st = _deg2rad(strike_deg); di = _deg2rad(dip_deg)
    u_s = np.array([ math.cos(st),  math.sin(st), 0.0 ])
    u_d = np.array([-math.cos(di)*math.sin(st),
                     math.cos(di)*math.cos(st),
                    -math.sin(di)])
    n = np.cross(u_s, u_d)
    for v in (u_s, u_d, n):
        nv = np.linalg.norm(v)
        if nv > 0: v[:] = v / nv
    return u_s, u_d, n

def _plane_normal(strike_deg: float, dip_deg: float) -> np.ndarray:
    _, _, n = _strike_dip_axes(strike_deg, dip_deg)
    return n

def _slip_vector(strike_deg: float, dip_deg: float, rake_deg: float) -> Optional[np.ndarray]:
    if any(pd.isna(v) for v in (strike_deg, dip_deg, rake_deg)):
        return None
    u_s, u_d, _ = _strike_dip_axes(strike_deg, dip_deg)
    ra = _deg2rad(rake_deg)
    v = math.cos(ra)*u_s + math.sin(ra)*u_d
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return None
    return v / nv

def _expected_slip_on_plane(strike_deg: float, dip_deg: float, conv_az_deg: float) -> Optional[np.ndarray]:
    az = _deg2rad(conv_az_deg)
    v_conv = np.array([math.cos(az), math.sin(az), 0.0])
    n = _plane_normal(strike_deg, dip_deg)
    v_proj = v_conv - np.dot(v_conv, n) * n
    nv = np.linalg.norm(v_proj)
    if nv < 1e-12:
        return None
    return v_proj / nv

def _geom_cone_ok(strike: float, dip: float, slab_strike: float, slab_dip: float, norm_cone_deg: float) -> bool:
    if any(pd.isna(v) for v in (strike, dip, slab_strike, slab_dip)):
        return False
    ang = _angle_between(_plane_normal(strike, dip), _plane_normal(slab_strike, slab_dip), oriented=False)
    return ang <= float(norm_cone_deg)

def _slip_cone_ok(strike: float, dip: float, rake: float, conv_az_deg: float, slip_cone_deg: float) -> bool:
    s_obs = _slip_vector(strike, dip, rake)
    s_exp = _expected_slip_on_plane(strike, dip, conv_az_deg)
    if (s_obs is None) or (s_exp is None):
        return False
    ang = _angle_between(s_obs, s_exp, oriented=True)
    return ang <= float(slip_cone_deg)

def interface_mech_ok_cone(row: pd.Series,
                           slab_strike: float, slab_dip: float,
                           conv_az_deg: float = CONV_AZIMUTH_DEG,
                           norm_cone_deg: float = NORM_CONE_DEG,
                           slip_cone_deg: float = SLIP_CONE_DEG) -> bool:
    s1, d1, r1 = row.get("strike1"), row.get("dip1"), row.get("rake1")
    s2, d2, r2 = row.get("strike2"), row.get("dip2"), row.get("rake2")

    ok = False
    if all(v is not None and not pd.isna(v) for v in (s1, d1, r1)):
        ok |= (_geom_cone_ok(s1, d1, slab_strike, slab_dip, norm_cone_deg)
               and _slip_cone_ok(s1, d1, r1, conv_az_deg, slip_cone_deg))
    if all(v is not None and not pd.isna(v) for v in (s2, d2, r2)):
        ok |= (_geom_cone_ok(s2, d2, slab_strike, slab_dip, norm_cone_deg)
               and _slip_cone_ok(s2, d2, r2, conv_az_deg, slip_cone_deg))
    return bool(ok)

# ---- Depth-based logic helpers ----
def _is_relocated(row: pd.Series) -> bool:
    for c in ("lon_error", "lat_error", "depth_error"):
        v = row.get(c)
        if isinstance(v, str) and v.strip().lower() == "relocated":
            return True
    return False

def _numeric_errors(row: pd.Series) -> tuple[float, float, float]:
    # Preserve strings like "relocated"; return NaN for non-numeric
    le = pd.to_numeric(pd.Series([row.get("lon_error")]), errors="coerce").iloc[0]
    la = pd.to_numeric(pd.Series([row.get("lat_error")]), errors="coerce").iloc[0]
    de = pd.to_numeric(pd.Series([row.get("depth_error")]), errors="coerce").iloc[0]
    return float(le) if pd.notna(le) else np.nan, float(la) if pd.notna(la) else np.nan, float(de) if pd.notna(de) else np.nan

def _ellipse_mask(lon_nodes: np.ndarray, lat_nodes: np.ndarray,
                  lon0: float, lat0: float, lon_err: float, lat_err: float) -> np.ndarray:
    if not (np.isfinite(lon_err) and np.isfinite(lat_err)) or (lon_err <= 0 and lat_err <= 0):
        return np.zeros_like(lon_nodes, dtype=bool)
    dx = (lon_nodes - lon0) / max(lon_err, 1e-12)
    dy = (lat_nodes - lat0) / max(lat_err, 1e-12)
    return (dx*dx + dy*dy) <= 1.0

# ---- Row classification (depth-first, then kinematics) ----
def classify_row(row: pd.Series, intra_poly, trench_line, slab: SlabGrid) -> tuple[str, float | None]:
    lon = pd.to_numeric(pd.Series([row.get("longitude")]), errors="coerce").iloc[0]
    lat = pd.to_numeric(pd.Series([row.get("latitude")]),  errors="coerce").iloc[0]
    dep = pd.to_numeric(pd.Series([row.get("depth")]),     errors="coerce").iloc[0]
    lon = float(lon) if pd.notna(lon) else np.nan
    lat = float(lat) if pd.notna(lat) else np.nan
    dep = float(dep) if pd.notna(dep) else np.nan

    slab_depth, slab_strike, slab_dip = query_slab(slab, lon, lat, SLAB_QUERY_MAXDIST_KM)

    # Domain logic unchanged for these:
    in_intra = (intra_poly is not None and np.isfinite(lon) and np.isfinite(lat)
                and intra_poly.contains(Point(lon, lat)))
    if in_intra and np.isfinite(dep):
        if slab_depth is None:
            if dep <= INTRA_ARC_SHALLOW_MAX: return "crustal_intraarc_shallow", np.nan
            if dep <= 90.0:                  return "crustal_intraarc_deep",   np.nan
            return "slab_deep", np.nan
        else:
            if dep >= slab_depth - DEEP_SLAB_TOL: return "slab_deep", slab_depth
            if dep <= INTRA_ARC_SHALLOW_MAX:      return "crustal_intraarc_shallow", np.nan
            return "crustal_intraarc_deep", np.nan

    if is_west_of_trench(lon, lat, trench_line): return "outer_rise", np.nan
    if (not in_intra) and is_east_of_polygon(lon, lat, intra_poly) and np.isfinite(dep) and dep < BACKARC_MAX_DEPTH:
        return "backarc", np.nan

    if (slab_depth is None) or (not np.isfinite(dep)): return "unclassified", np.nan
    if slab_depth > SUBDUCTION_CLASSIFY_MAX_SLAB_DEPTH: return "slab_deep", slab_depth

    # ---- DEPTH-FIRST classification in subduction domain ----
    relocated = _is_relocated(row)
    lon_err, lat_err, dep_err = _numeric_errors(row)

    # 1) Relocated → strict small buffer
    if relocated:
        if dep <= slab_depth - INTERFACE_DEPTH_TOL: return "forearc", slab_depth
        if dep >= slab_depth + INTERFACE_DEPTH_TOL: return "intra_slab", slab_depth
        # ambiguous (inside buffer) → go to kinematics

    else:
        # 2) Numeric errors → ellipse sampling of slab depths
        if np.isfinite(dep_err) and dep_err > 0:
            mask = _ellipse_mask(slab.lon, slab.lat, lon, lat, lon_err, lat_err)
            if mask.any():
                min_slab = float(np.nanmin(slab.depth[mask]))
                max_slab = float(np.nanmax(slab.depth[mask]))
                if np.isfinite(min_slab) and (dep + dep_err < min_slab): return "forearc", slab_depth
                if np.isfinite(max_slab) and (dep - dep_err > max_slab): return "intra_slab", slab_depth
            # ambiguous → continue to kinematics

        # 3) Explicit zero error → poorly constrained → strict buffer
        elif np.isfinite(dep_err) and dep_err == 0.0:
            if dep <= slab_depth - STRICT_INTERFACE_DEPTH_TOL: return "forearc", slab_depth
            if dep >= slab_depth + STRICT_INTERFACE_DEPTH_TOL: return "intra_slab", slab_depth
            # ambiguous → continue to kinematics

        # 4) No error info at all → default depth gate
        else:
            if dep <= slab_depth - INTERFACE_DEPTH_TOL: return "forearc", slab_depth
            if dep >= slab_depth + INTERFACE_DEPTH_TOL: return "intra_slab", slab_depth
            # ambiguous → continue to kinematics

    # ---- ONLY ambiguous events in the buffer reach kinematics ----
    if interface_mech_ok_cone(row, slab_strike=slab_strike, slab_dip=slab_dip,
                              conv_az_deg=CONV_AZIMUTH_DEG,
                              norm_cone_deg=NORM_CONE_DEG,
                              slip_cone_deg=SLIP_CONE_DEG):
        return "slab_interface", slab_depth
    return ("forearc" if dep < slab_depth else "intra_slab"), slab_depth

# ---- Catalog driver ----
def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

def classify_catalog(in_csv: str, out_csv: str,
                     intra_poly, trench_line, slab: SlabGrid) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df = pd.read_csv(in_csv)

    # A) DO NOT coerce lon_error/lat_error/depth_error — we need the string "relocated"
    numeric_cols = [
        "longitude","latitude","depth",
        "strike1","dip1","rake1","strike2","dip2","rake2",
        "Mrr","Mtt","Mpp","Mrt","Mrp","Mtp",
    ]
    _coerce_numeric(df, numeric_cols)

    classes = []
    subdeps = []
    for _, row in df.iterrows():
        cls, sd = classify_row(row, intra_poly, trench_line, slab)
        classes.append(cls)
        subdeps.append(sd)
    df["class"] = classes
    df["sub_depth"] = subdeps
    print("Class totals: " + ", ".join(f"{c}={int((df['class'] == c).sum())}" for c in CLASSES))
    df.to_csv(out_csv, index=False)
    print(f"[OK] {out_csv} ({len(df)} rows)")

def main() -> None:
    intra_poly  = load_union(str(paths.intraarc_shp))
    trench_line = load_union(str(paths.trench_shp))
    slab        = load_slab_xyz(str(paths.slab_depth), str(paths.slab_strike), str(paths.slab_dip))

    # classify the relocated merged catalog
    classify_catalog(
        in_csv=str(paths.cat_relocated),
        out_csv=str(paths.relocated_classified),
        intra_poly=intra_poly,
        trench_line=trench_line,
        slab=slab,
    )
    classify_catalog(
        in_csv=str(paths.cat_merged),
        out_csv=str(paths.merged_classified),
        intra_poly=intra_poly,
        trench_line=trench_line,
        slab=slab,
    )
if __name__ == "__main__":
    main()
