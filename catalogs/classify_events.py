#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple

from shapely.geometry import Point
from shapely.ops import nearest_points
import geopandas as gpd
from scipy.spatial import cKDTree

# --- replace these at the top ---
INTRA_ARC_MAX_DEPTH   = 40.0   # km
DEEP_EQ_DEPTH         = 100.0  # km  (everything deeper is "deep")

# -------- TUNABLES --------
INTRA_ARC_MAX_DEPTH      = 40.0   # km
DEEP_EQ_DEPTH         = 100.0  # km  (everything deeper is "deep")

FOREARC_MAX_DEPTH        = 60.0   # km
SLAB_ABOVE_MARGIN        = 5.0    # km

INTERFACE_DEPTH_TOL      = 5.0   # km
STRIKE_TOL               = 25.0   # deg (mod 180)
DIP_TOL                  = 20.0   # deg

SLAB_QUERY_MAXDIST_KM    = 15.0   # ignore slab if nearest grid node farther than this


# ------------------ INPUT PATHS ------------------
CATALOG_CSV   = "merged_all_catalogs.csv"  # input catalog (fields: id,time_iso,longitude,latitude,depth,mag,...)
INTRA_ARC_SHP = "../polygons/intraarc_polygon.shp"            # polygon shapefile
TRENCH_LINE_SHP = "../shapefiles/chile_trench.shp"        # line shapefile

SLAB_DEPTH_XYZ = "../slab2/sam_slab2_dep_02.23.18.xyz"    # lon(0-360),lat,depth(km or NaN)
SLAB_STRIKE_XYZ= "../slab2/sam_slab2_str_02.23.18.xyz"   # lon(0-360),lat,strike(0-360 or NaN)
SLAB_DIP_XYZ   = "../slab2/sam_slab2_dip_02.23.18.xyz"      # lon(0-360),lat,dip(0-90 or NaN)

# -------- OUTPUT --------
OUT_DIR         = "classified_catalogs"     # folder for class CSVs
WRITE_COMBINED  = True
COMBINED_CSV    = "catalog_labeled.csv"

CLASSES = [
    "intra_arc_shallow",
    "intra_arc_deep",
    "subduction_interface",
    "subduction_intraslab",
    "deep",                 # <-- new
    "outer_rise",
    "forearc",
    "unclassified",
]
# -------- helpers --------
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

def strike_diff(a, b):
    # strike periodicity 180°
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)

def dip_diff(a, b):
    return abs(a - b)

# -------- slab model --------
@dataclass
class SlabModel:
    pts_lonlat: np.ndarray   # Nx2 (lon,lat)
    depth: np.ndarray        # N
    strike: np.ndarray       # N
    dip: np.ndarray          # N
    kdtree: cKDTree

    @classmethod
    def from_xyz(cls, depth_path, strike_path, dip_path):
        def load_xyz(path):
            df = pd.read_csv(path, header=None, names=["x","y","v"])
            lon = df["x"].astype(float).map(lon_0360_to_180).values
            lat = df["y"].astype(float).values
            val = pd.to_numeric(df["v"], errors="coerce").values
            return lon, lat, val

        lon_d, lat_d, dep = load_xyz(depth_path)
        dep = -dep
        lon_s, lat_s, stk = load_xyz(strike_path)
        lon_i, lat_i, dip = load_xyz(dip_path)

        if not (len(lon_d)==len(lon_s)==len(lon_i)
                and np.allclose(lon_d, lon_s) and np.allclose(lat_d, lat_s)
                and np.allclose(lon_d, lon_i) and np.allclose(lat_d, lat_i)):
            raise ValueError("Slab XYZ files are not aligned on identical lon/lat grids.")

        pts = np.column_stack([lon_d, lat_d])
        kdt = cKDTree(pts)
        return cls(pts, dep, stk, dip, kdt)

    def query(self, lon, lat) -> Tuple[Optional[float], Optional[float], Optional[float], float]:
        if lon is None or lat is None or np.isnan(lon) or np.isnan(lat):
            return None, None, None, float("inf")
        d2, idx = self.kdtree.query([lon, lat], k=1)
        nn_lon, nn_lat = self.pts_lonlat[idx]
        dist_km = haversine_km(lon, lat, float(nn_lon), float(nn_lat))
        dep = float(self.depth[idx]) if np.isfinite(self.depth[idx]) else None
        stk = float(self.strike[idx]) if np.isfinite(self.strike[idx]) else None
        dp  = float(self.dip[idx])    if np.isfinite(self.dip[idx])    else None
        return dep, stk, dp, dist_km

# -------- geometry I/O --------
def load_intra_arc_polygon(path):
    gdf = gpd.read_file(path)
    if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs(4326)
    return gdf.unary_union  # Polygon/MultiPolygon

def load_trench_line(path):
    gdf = gpd.read_file(path)
    if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs(4326)
    return gdf.unary_union  # LineString/MultiLineString

def is_inside(poly_geom, lon, lat) -> bool:
    if poly_geom is None or np.isnan(lon) or np.isnan(lat):
        return False
    return poly_geom.contains(Point(float(lon), float(lat)))

def is_west_of_line(line_geom, lon, lat) -> bool:
    """
    Approximate 'west of trench': compare lon to the lon of nearest point on the trench line.
    Works reasonably for a mostly N–S trench.
    """
    if line_geom is None or np.isnan(lon) or np.isnan(lat):
        return False
    p = Point(float(lon), float(lat))
    np_line, _ = nearest_points(line_geom, p)
    return lon < np_line.x


def classify_row(row, slab: SlabModel, intra_poly, trench_line) -> str:
    lon = row["longitude"]; lat = row["latitude"]; dep = row["depth"]
    s1, d1 = row.get("strike1"), row.get("dip1")
    s2, d2 = row.get("strike2"), row.get("dip2")

    # 1) Intra-arc first
    inside_arc = pd.notna(lon) and pd.notna(lat) and is_inside(intra_poly, lon, lat)
    if inside_arc and pd.notna(dep):
        if dep <= INTRA_ARC_MAX_DEPTH:
            return "intra_arc_shallow"
        if INTRA_ARC_MAX_DEPTH < dep < DEEP_EQ_DEPTH:
            return "intra_arc_deep"
        # dep >= DEEP_EQ_DEPTH → handled later as "deep"

    # Slab at epicenter
    slab_dep, slab_strike, slab_dip, slab_dist = slab.query(lon, lat)
    slab_ok = (slab_dep is not None) and (slab_dist <= SLAB_QUERY_MAXDIST_KM)

    # 2) Subduction interface: near-slab depth AND plane ~ slab strike/dip
    if slab_ok and pd.notna(dep) and abs(dep - slab_dep) <= INTERFACE_DEPTH_TOL:
        def plane_matches(strike, dip):
            if pd.isna(strike) or pd.isna(dip) or slab_strike is None or slab_dip is None:
                return False
            return (strike_diff(float(strike), float(slab_strike)) <= STRIKE_TOL and
                    dip_diff(float(dip), float(slab_dip)) <= DIP_TOL)

        print('AAAA')

        if plane_matches(s1, d1) or plane_matches(s2, d2):
            return "subduction_interface"

    # 3) Subduction intraslab: beneath slab by tolerance (no upper depth cap now)
    if slab_ok and pd.notna(dep) and (dep > slab_dep + INTERFACE_DEPTH_TOL):
        return "subduction_intraslab"

    # 4) Deep earthquakes (>= DEEP_EQ_DEPTH), if still unlabeled
    if pd.notna(dep) and dep >= DEEP_EQ_DEPTH:
        return "deep"

    # 5) Outer-rise
    if pd.notna(lon) and pd.notna(lat) and is_west_of_line(trench_line, lon, lat):
        return "outer_rise"

    # 6) Forearc (remaining, shallow-ish)
    if pd.notna(dep) and dep <= FOREARC_MAX_DEPTH:
        cond_above_slab = slab_ok and (dep < slab_dep - SLAB_ABOVE_MARGIN)
        cond_no_slab_outside_arc = (not slab_ok) and (not inside_arc)
        if cond_above_slab or cond_no_slab_outside_arc:
            return "forearc"

    # 7) Unclassified
    return "unclassified"


# -------- main --------
if __name__ == '__main__':
    # Read catalog
    df = pd.read_csv(CATALOG_CSV)
    for col in ["longitude","latitude","depth","strike1","dip1","strike2","dip2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Load geometries + slab
    intra_poly  = load_intra_arc_polygon(INTRA_ARC_SHP)
    trench_line = load_trench_line(TRENCH_LINE_SHP)
    slab        = SlabModel.from_xyz(SLAB_DEPTH_XYZ, SLAB_STRIKE_XYZ, SLAB_DIP_XYZ)

    # Classify
    df["class"] = [classify_row(r, slab, intra_poly, trench_line) for _, r in df.iterrows()]

    # Optional combined file
    if WRITE_COMBINED:
        df.to_csv(COMBINED_CSV, index=False)

    # Per-class outputs
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    for cl in CLASSES:
        part = df[df["class"] == cl].copy()
        part.to_csv(os.path.join(OUT_DIR, f"{cl}.csv"), index=False)

    # Summary
    print("Counts by class:")
    print(df["class"].value_counts().reindex(CLASSES, fill_value=0))
    print(f"\nWrote class CSVs under: {OUT_DIR}")
    if WRITE_COMBINED:
        print(f"Combined labeled catalog: {COMBINED_CSV}")

