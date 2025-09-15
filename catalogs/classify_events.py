#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classifier for Andean subduction (Chile) — MECH checks disabled for now.

Rules (in order):
1) Inside intra-arc polygon:
   (a) If NO slab nearby:
       - depth <= INTRA_ARC_SHALLOW_MAX          -> crustal_intraarc_shallow
       - INTRA_ARC_SHALLOW_MAX < depth <= 90 km  -> crustal_intraarc_deep
       - depth > 90 km                            -> slab_deep
   (b) If slab is defined:
       - depth >= slab_depth + DEEP_SLAB_TOL      -> slab_deep
       - depth <= INTRA_ARC_SHALLOW_MAX           -> crustal_intraarc_shallow
       - else                                     -> crustal_intraarc_deep
2) WEST (oceanward) of trench line                -> outer_rise
3) EAST of intra-arc polygon AND depth < 50 km    -> backarc
4) Subduction domain (slab defined AND slab_depth ≤ 70 km):
   - depth <= slab_depth - INTERFACE_DEPTH_TOL    -> forearc
   - |depth - slab_depth| <= INTERFACE_DEPTH_TOL  -> slab_interface
   - depth >= slab_depth + INTERFACE_DEPTH_TOL    -> intra_slab
5) slab_depth > 70 km (and slab defined)          -> slab_deep
6) Otherwise                                      -> unclassified
"""

import os
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from shapely.geometry import Point
from shapely.ops import unary_union
import geopandas as gpd
from scipy.spatial import cKDTree

# ------------------ TUNABLES ------------------
INTRA_ARC_SHALLOW_MAX = 32.0    # km
DEEP_SLAB_TOL         = 10.0    # km (inside intra-arc, clearly below slab => slab_deep)

SUBDUCTION_CLASSIFY_MAX_SLAB_DEPTH = 70.0  # km
INTERFACE_DEPTH_TOL    = 5.0     # km (3–5 km recommended)
BACKARC_MAX_DEPTH      = 50.0    # km -> backarc requires depth < 50 km

SLAB_QUERY_MAXDIST_KM  = 15.0    # nearest slab node must be within this distance
SLAB_DEPTH_IS_POSITIVE_DOWN = False  # True if slab2 depth values are +down; set False to flip on load

# ------------------ CLASSES ------------------
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

# ------------------ INPUTS ------------------
PROCESSED_DIR = "processed_catalogs"
INPUT_FILES: Dict[str, str] = {
    "gcmt":     os.path.join(PROCESSED_DIR, "gcmt_formatted.csv"),
    "anss":     os.path.join(PROCESSED_DIR, "anss_formatted.csv"),
    "isc":      os.path.join(PROCESSED_DIR, "isc_formatted.csv"),
    "isc_gem":  os.path.join(PROCESSED_DIR, "isc_gem_formatted.csv"),
    "gmt_nico": os.path.join(PROCESSED_DIR, "gmt_nico_formatted.csv"),
    "merged":   os.path.join(PROCESSED_DIR, "merged_catalog.csv"),
    "full":     os.path.join(PROCESSED_DIR, "full_catalog_with_dups.csv"),
}

INTRA_ARC_SHP   = "../polygons/intraarc_polygon.shp"      # EPSG:4326
TRENCH_LINE_SHP = "../shapefiles/chile_trench.shp"        # EPSG:4326
SLAB_DEPTH_XYZ  = "../slab2/sam_slab2_dep_02.23.18.xyz"   # lon(0..360), lat, depth (+down if flag True)
SLAB_STRIKE_XYZ = "../slab2/sam_slab2_str_02.23.18.xyz"
SLAB_DIP_XYZ    = "../slab2/sam_slab2_dip_02.23.18.xyz"

OUT_ROOT = "classified_catalogs"

# ------------------ helpers ------------------
def lon_0360_to_180(lon):
    lon = float(lon)
    return lon - 360.0 if lon > 180.0 else lon

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl   = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

# ------------------ slab grid ------------------
@dataclass
class SlabGrid:
    tree: cKDTree
    lon: np.ndarray
    lat: np.ndarray
    depth: np.ndarray   # km, +down
    strike: np.ndarray  # deg
    dip: np.ndarray     # deg

def _read_xyz(p):
    arr = pd.read_csv(p, header=None, names=["lon","lat","val"])
    arr["lon"] = arr["lon"].astype(float).apply(lon_0360_to_180)
    arr["lat"] = arr["lat"].astype(float)
    return arr

def load_slab_xyz(depth_path: str, strike_path: str, dip_path: str) -> SlabGrid:
    dep = _read_xyz(depth_path)
    st  = _read_xyz(strike_path)
    di  = _read_xyz(dip_path)

    if not SLAB_DEPTH_IS_POSITIVE_DOWN:
        dep["val"] = -dep["val"]

    merged = dep.merge(st, on=["lon","lat"], how="outer", suffixes=("_dep","_st"))
    merged = merged.merge(di, on=["lon","lat"], how="outer")
    merged.rename(columns={"val":"val_dip"}, inplace=True)
    merged = merged.dropna(subset=["val_dep","val_st","val_dip"], how="all")

    lon = merged["lon"].to_numpy(float)
    lat = merged["lat"].to_numpy(float)
    depth = merged["val_dep"].to_numpy(float)         # keep +down
    strike = merged["val_st"].to_numpy(float)
    dip = merged["val_dip"].to_numpy(float)

    tree = cKDTree(np.c_[lon, lat])
    return SlabGrid(tree=tree, lon=lon, lat=lat, depth=depth, strike=strike, dip=dip)

def query_slab(grid: SlabGrid, lon: float, lat: float, maxdist_km: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if np.isnan(lon) or np.isnan(lat):
        return (None, None, None)
    dist, idx = grid.tree.query(np.array([lon, lat]), k=1)
    if np.isinf(dist) or idx is None:
        return (None, None, None)
    dkm = haversine_km(lon, lat, grid.lon[idx], grid.lat[idx])
    if dkm > maxdist_km:
        return (None, None, None)
    dep = grid.depth[idx]
    st  = grid.strike[idx]
    di  = grid.dip[idx]
    dep_val = None if pd.isna(dep) else float(dep)
    st_val  = None if pd.isna(st)  else float(st % 360.0)
    di_val  = None if pd.isna(di)  else float(di)
    return (dep_val, st_val, di_val)

# ------------------ geometry helpers ------------------
def load_intra_arc_polygon(path: str):
    if not os.path.exists(path):
        return None
    gdf = gpd.read_file(path).to_crs(epsg=4326)
    return unary_union(gdf.geometry)

def load_trench_line(path: str):
    if not os.path.exists(path):
        return None
    gdf = gpd.read_file(path).to_crs(epsg=4326)
    return unary_union(gdf.geometry)

def is_west_of_trench(lon: float, lat: float, trench_line) -> bool:
    if trench_line is None or np.isnan(lon) or np.isnan(lat):
        return False
    pt = Point(lon, lat)
    nearest = trench_line.interpolate(trench_line.project(pt))
    return float(lon) < float(nearest.x)  # further west = more negative lon

def is_east_of_polygon(lon: float, lat: float, polygon) -> bool:
    if polygon is None or np.isnan(lon) or np.isnan(lat):
        return False
    boundary = polygon.boundary
    pt = Point(lon, lat)
    nearest = boundary.interpolate(boundary.project(pt))
    return float(lon) > float(nearest.x)

# ------------------ classification ------------------
def classify_row(row, intra_poly, trench_line, slab: SlabGrid) -> str:
    lon = float(row.get("longitude")) if row.get("longitude") is not None else np.nan
    lat = float(row.get("latitude"))  if row.get("latitude")  is not None else np.nan
    dep = float(row.get("depth"))     if row.get("depth")     is not None else np.nan

    slab_depth, slab_strike, slab_dip = query_slab(slab, lon, lat, SLAB_QUERY_MAXDIST_KM)

    # 1) Inside intra-arc polygon
    in_intra = False
    if intra_poly is not None and not (np.isnan(lon) or np.isnan(lat)):
        in_intra = intra_poly.contains(Point(lon, lat))

    if in_intra and not np.isnan(dep):
        if slab_depth is None:
            # new rule: tiered by depth
            if dep <= INTRA_ARC_SHALLOW_MAX:
                return "crustal_intraarc_shallow"
            elif dep <= 90.0:
                return "crustal_intraarc_deep"
            else:
                return "slab_deep"
        else:
            # slab is available
            if dep >= slab_depth - DEEP_SLAB_TOL:
                return "slab_deep"
            if dep <= INTRA_ARC_SHALLOW_MAX:
                return "crustal_intraarc_shallow"
            else:
                return "crustal_intraarc_deep"

    # 2) WEST (oceanward) of trench -> outer_rise
    if is_west_of_trench(lon, lat, trench_line):
        return "outer_rise"

    # 3) EAST of intra-arc polygon AND shallow (< 50 km) -> backarc
    if (not in_intra) and is_east_of_polygon(lon, lat, intra_poly) and (not np.isnan(dep)) and (dep < BACKARC_MAX_DEPTH):
        return "backarc"

    # 4) Subduction domain (slab defined)
    if (slab_depth is not None) and (not np.isnan(dep)):
        if slab_depth <= SUBDUCTION_CLASSIFY_MAX_SLAB_DEPTH:
            if dep <= slab_depth - INTERFACE_DEPTH_TOL:
                return "forearc"
            elif abs(dep - slab_depth) <= INTERFACE_DEPTH_TOL:
                return "slab_interface"
            elif dep >= slab_depth + INTERFACE_DEPTH_TOL:
                return "intra_slab"
        else:
            return "slab_deep"

    # 5) Fallback
    return "unclassified"

# ------------------ run on one CSV ------------------
def classify_catalog(csv_path: str, out_folder: str,
                     intra_poly, trench_line, slab: SlabGrid) -> None:
    os.makedirs(out_folder, exist_ok=True)
    df = pd.read_csv(csv_path)

    for col in ["longitude","latitude","depth","strike1","dip1","rake1","strike2","dip2","rake2",
                "T_plunge","T_azimuth","N_plunge","N_azimuth","P_plunge","P_azimuth",
                "Mrr","Mtt","Mpp","Mrt","Mrp","Mtp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["class"] = [classify_row(row, intra_poly, trench_line, slab) for _, row in df.iterrows()]

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    combined_path = os.path.join(out_folder, f"{base_name}_classified.csv")
    df.to_csv(combined_path, index=False)
    print(f"[OK] {combined_path}  ({len(df)} rows)")

    for cls in CLASSES:
        sub = df[df["class"] == cls].copy()
        sub_path = os.path.join(out_folder, f"{cls}.csv")
        sub.to_csv(sub_path, index=False)

# ------------------ main ------------------
def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    intra_poly = load_intra_arc_polygon(INTRA_ARC_SHP)
    trench_line = load_trench_line(TRENCH_LINE_SHP)
    slab = load_slab_xyz(SLAB_DEPTH_XYZ, SLAB_STRIKE_XYZ, SLAB_DIP_XYZ)

    for name, path in INPUT_FILES.items():
        if not os.path.exists(path):
            print(f"[skip] not found: {path}")
            continue
        out_folder = os.path.join(OUT_ROOT, name)
        classify_catalog(path, out_folder, intra_poly, trench_line, slab)

if __name__ == "__main__":
    main()
