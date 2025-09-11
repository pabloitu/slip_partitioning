#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classifier (placeholder MECH checks) for Andean subduction zone (Chile).

- Uses fixed convergence azimuth = 78° (but mechanism checks are currently relaxed/placeholder).
- Outer-rise: west of trench polyline.
- Intra-arc shallow/deep by polygon and depth thresholds.
- Subduction domain logic by proximity to slab surface and depth windows.
- Writes: one classified CSV per catalog + per-class CSVs, under classified_catalogs/<catalog_name>/.

NOTE: Slab depth sign — set SLAB_DEPTH_IS_POSITIVE_DOWN accordingly.
"""

import os
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

from shapely.geometry import Point
import geopandas as gpd
from scipy.spatial import cKDTree

# ------------------ TUNABLES ------------------
# Intra-arc
INTRA_ARC_SHALLOW_MAX = 32.0
INTRA_ARC_DEEP_MARGIN = 10.0

# Subduction zoning
SUBDUCTION_CLASSIFY_MAX_SLAB_DEPTH = 70.0
INTERFACE_DEPTH_TOL = 7.0
FOREARC_MAX_DEPTH   = 70.0
SLAB_ABOVE_MARGIN   = 5.0

# Convergence (placeholder – mechanism checks relaxed)
CONV_AZIMUTH_DEG = 78.0

# Slab query
SLAB_QUERY_MAXDIST_KM = 15.0
SLAB_DEPTH_IS_POSITIVE_DOWN = True  # set False if your .xyz has negative depth for +down

# ------------------ CLASSES ------------------
CLASSES = [
    "crustal_intraarc_shallow",
    "crustal_intraarc_deep",
    "subduction_interface",
    "subduction_intraslab",
    "deep_subduction",
    "outer_rise",
    "forearc",
    "deep",            # intra-arc but beneath slab
    "unclassified",
]

# ------------------ INPUTS (edit paths here) ------------------
PROCESSED_DIR = "processed_catalogs"
INPUT_FILES: Dict[str, str] = {
    "gcmt": os.path.join(PROCESSED_DIR, "gcmt_formatted.csv"),
    "anss": os.path.join(PROCESSED_DIR, "anss_formatted.csv"),
    "isc":  os.path.join(PROCESSED_DIR, "isc_formatted.csv"),
    "isc_gem": os.path.join(PROCESSED_DIR, "isc_gem_formatted.csv"),
    "gmt_nico": os.path.join(PROCESSED_DIR, "gmt_nico_formatted.csv"),
    "merged": os.path.join(PROCESSED_DIR, "merged_catalog.csv"),
    "full":   os.path.join(PROCESSED_DIR, "full_catalog_with_dups.csv"),
}

INTRA_ARC_SHP   = "../polygons/intraarc_polygon.shp"      # EPSG:4326
TRENCH_LINE_SHP = "../shapefiles/chile_trench.shp"        # EPSG:4326
SLAB_DEPTH_XYZ  = "../slab2/sam_slab2_dep_02.23.18.xyz"   # lon(0..360), lat, depth (+down if SLAB_DEPTH_IS_POSITIVE_DOWN)
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

def angdiff_line(a: float, b: float) -> float:
    return abs(((a - b + 90.0) % 180.0) - 90.0)

def slip_azimuth_from_sdr(strike_deg, dip_deg, rake_deg) -> Optional[float]:
    if any(pd.isna(v) for v in [strike_deg, dip_deg, rake_deg]):
        return None
    strike = math.radians(float(strike_deg))
    dip    = math.radians(float(dip_deg))
    rake   = math.radians(float(rake_deg))
    u_s = np.array([math.cos(strike), math.sin(strike), 0.0])
    u_d = np.array([-math.cos(dip)*math.sin(strike), math.cos(dip)*math.cos(strike), -math.sin(dip)])
    s_vec = math.cos(rake) * u_s + math.sin(rake) * u_d
    sN, sE = float(s_vec[0]), float(s_vec[1])
    if abs(sN) < 1e-12 and abs(sE) < 1e-12:
        return None
    return (math.degrees(math.atan2(sE, sN)) + 360.0) % 360.0

# ------------------ slab grid ------------------
@dataclass
class SlabGrid:
    tree: cKDTree
    lon: np.ndarray
    lat: np.ndarray
    depth: np.ndarray
    strike: np.ndarray
    dip: np.ndarray

def load_slab_xyz(depth_path: str, strike_path: str, dip_path: str) -> SlabGrid:
    def read_xyz(p):
        arr = pd.read_csv(p, header=None, names=["lon","lat","val"])
        arr["lon"] = arr["lon"].astype(float).apply(lon_0360_to_180)
        arr["lat"] = arr["lat"].astype(float)
        return arr

    dep = read_xyz(depth_path)
    if not SLAB_DEPTH_IS_POSITIVE_DOWN:
        dep["val"] = -dep["val"]  # flip if files are negative for +down
    st  = read_xyz(strike_path)
    di  = read_xyz(dip_path)

    merged = dep.merge(st, on=["lon","lat"], how="outer", suffixes=("_dep","_st"))
    merged = merged.merge(di, on=["lon","lat"], how="outer")
    merged.rename(columns={"val":"val_dip"}, inplace=True)
    merged = merged.dropna(subset=["val_dep","val_st","val_dip"], how="all")

    lon = merged["lon"].to_numpy(float)
    lat = merged["lat"].to_numpy(float)
    depth = merged["val_dep"].to_numpy(float)
    strike = merged["val_st"].to_numpy(float)
    dip = merged["val_dip"].to_numpy(float)

    coords = np.c_[lon, lat]
    tree = cKDTree(coords)
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
    return gdf.unary_union

def load_trench_line(path: str):
    if not os.path.exists(path):
        return None
    gdf = gpd.read_file(path).to_crs(epsg=4326)
    return gdf.unary_union

def is_west_of_trench(lon: float, lat: float, trench_line) -> bool:
    if trench_line is None or np.isnan(lon) or np.isnan(lat):
        return False
    pt = Point(lon, lat)
    nearest = trench_line.interpolate(trench_line.project(pt))
    return float(lon) < float(nearest.x)

# ------------------ placeholder mechanism test ------------------
def nodal_plane_matches_interface(strike: float, dip: float, rake: float,
                                  slab_strike: Optional[float], slab_dip: Optional[float],
                                  p_azimuth: Optional[float]) -> bool:
    """
    Placeholder: returns True (mechanism filters disabled for now).
    Keep function shape so we can tighten it later.
    """
    # saz = slip_azimuth_from_sdr(strike, dip, rake)
    # if saz is None or angdiff_line(saz, CONV_AZIMUTH_DEG) > 45.0: return False
    # (other tests off)
    return True

def is_interface_mechanism(row, slab_strike: Optional[float], slab_dip: Optional[float]) -> bool:
    s1, d1, r1 = row.get("strike1"), row.get("dip1"), row.get("rake1")
    s2, d2, r2 = row.get("strike2"), row.get("dip2"), row.get("rake2")
    p_az = row.get("P_azimuth")
    ok1 = (s1 is not None and d1 is not None and r1 is not None) and \
          nodal_plane_matches_interface(s1, d1, r1, slab_strike, slab_dip, p_az)
    if ok1: return True
    ok2 = (s2 is not None and d2 is not None and r2 is not None) and \
          nodal_plane_matches_interface(s2, d2, r2, slab_strike, slab_dip, p_az)
    return bool(ok2)

# ------------------ classification ------------------
def classify_row(row, intra_poly, trench_line, slab: SlabGrid) -> str:
    lon = float(row.get("longitude")) if row.get("longitude") is not None else np.nan
    lat = float(row.get("latitude"))  if row.get("latitude")  is not None else np.nan
    dep = float(row.get("depth"))     if row.get("depth")     is not None else np.nan

    slab_depth, slab_strike, slab_dip = query_slab(slab, lon, lat, SLAB_QUERY_MAXDIST_KM)

    in_intra = False
    if intra_poly is not None and not (np.isnan(lon) or np.isnan(lat)):
        in_intra = intra_poly.contains(Point(lon, lat))

    west_of_tr = is_west_of_trench(lon, lat, trench_line)

    # 1) Outer-rise
    if west_of_tr:
        return "outer_rise"

    # 2) Intra-arc shallow
    if in_intra and not np.isnan(dep) and dep <= INTRA_ARC_SHALLOW_MAX:
        return "crustal_intraarc_shallow"

    # 3) Slab-defined logic
    if slab_depth is not None and not np.isnan(dep):
        if slab_depth <= SUBDUCTION_CLASSIFY_MAX_SLAB_DEPTH:
            depth_diff = abs(dep - slab_depth)

            # interface
            if depth_diff <= INTERFACE_DEPTH_TOL and is_interface_mechanism(row, slab_strike, slab_dip):
                return "subduction_interface"

            # intraslab
            if dep >= slab_depth + INTERFACE_DEPTH_TOL:
                return "subduction_intraslab"

            # forearc (above slab and shallow, not intra-arc)
            if (dep <= slab_depth - SLAB_ABOVE_MARGIN) and (dep <= FOREARC_MAX_DEPTH) and (not in_intra):
                return "forearc"

            # intra-arc deep (above slab but deeper than shallow threshold)
            if in_intra and (dep > INTRA_ARC_SHALLOW_MAX) and (dep <= slab_depth + INTRA_ARC_DEEP_MARGIN):
                return "crustal_intraarc_deep"

            # intra-arc but beneath slab
            if in_intra and (dep >= slab_depth + INTERFACE_DEPTH_TOL):
                return "deep"

        else:
            # deep subduction domain
            if (abs(dep - slab_depth) <= INTERFACE_DEPTH_TOL) or (dep >= slab_depth - INTERFACE_DEPTH_TOL):
                return "deep_subduction"

    # 4) Intra-arc deep (no slab or outside shallow domain)
    if in_intra and not np.isnan(dep) and dep > INTRA_ARC_SHALLOW_MAX:
        return "crustal_intraarc_deep"

    # 5) Forearc shallow (no slab info)
    if not in_intra and not np.isnan(dep) and dep <= FOREARC_MAX_DEPTH:
        return "forearc"

    return "unclassified"

# ------------------ run on one CSV ------------------
def classify_catalog(csv_path: str, out_folder: str,
                     intra_poly, trench_line, slab: SlabGrid) -> None:
    os.makedirs(out_folder, exist_ok=True)
    df = pd.read_csv(csv_path)

    # normalize numeric columns if present
    for col in ["longitude","latitude","depth","strike1","dip1","rake1","strike2","dip2","rake2",
                "T_plunge","T_azimuth","N_plunge","N_azimuth","P_plunge","P_azimuth",
                "Mrr","Mtt","Mpp","Mrt","Mrp","Mtp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    labels = []
    for _, row in df.iterrows():
        labels.append(classify_row(row, intra_poly, trench_line, slab))
    df["class"] = labels

    # write combined
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    combined_path = os.path.join(out_folder, f"{base_name}_classified.csv")
    df.to_csv(combined_path, index=False)
    print(f"[OK] {combined_path}  ({len(df)} rows)")

    # write per-class
    for cls in CLASSES:
        sub = df[df["class"] == cls].copy()
        sub_path = os.path.join(out_folder, f"{cls}.csv")
        sub.to_csv(sub_path, index=False)

# ------------------ main ------------------
def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    # load spatial data (once)
    intra_poly = load_intra_arc_polygon(INTRA_ARC_SHP)
    trench_line = load_trench_line(TRENCH_LINE_SHP)
    slab = load_slab_xyz(SLAB_DEPTH_XYZ, SLAB_STRIKE_XYZ, SLAB_DIP_XYZ)

    # classify each catalog into its own subfolder
    for name, path in INPUT_FILES.items():
        if not os.path.exists(path):
            print(f"[skip] not found: {path}")
            continue
        out_folder = os.path.join(OUT_ROOT, name)
        classify_catalog(path, out_folder, intra_poly, trench_line, slab)

if __name__ == "__main__":
    main()
