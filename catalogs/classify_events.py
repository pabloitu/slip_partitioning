#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Event classifier for Andean subduction zone (Chile) using a fixed convergence trend.

Key points:
- Convergence trend is fixed to N78°E (78° azimuth from North, clockwise).
- Subduction-interface test (only where slab depth <= 70 km) requires:
    • |event_depth - slab_depth| <= 10 km
    • nodal-plane slip azimuth ~ 78° (within CONV_AZ_TOL)
    • optional: P-axis azimuth ~ 78° (within P_AXIS_AZ_TOL) and P plunge <= P_AXIS_MAX_PLUNGE
    • optional: slab strike/dip compatibility
- Other classes retained:
    outer_rise, crustal_intraarc_shallow, crustal_intraarc_deep,
    subduction_intraslab, deep_subduction, forearc, deep (intra-arc but beneath slab), unclassified
- Outputs combined CSV + one CSV per class.

Inputs:
  - Catalog CSV columns (at least):
      id, time_iso, longitude, latitude, depth, mag, mag_type,
      strike1, dip1, rake1, strike2, dip2, rake2,
      T_plunge, T_azimuth, N_plunge, N_azimuth, P_plunge, P_azimuth, ...
  - Intra-arc polygon shapefile (EPSG:4326)
  - Trench polyline shapefile (EPSG:4326)
  - Slab2.0 XYZ files: depth (km, +down, NaN where undefined), strike (deg), dip (deg)
"""

import os
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple

from shapely.geometry import Point
import geopandas as gpd
from scipy.spatial import cKDTree

# ------------------ TUNABLES ------------------
# Intra-arc buckets
INTRA_ARC_SHALLOW_MAX = 32.0   # km (crustal intra-arc shallow upper bound)
INTRA_ARC_DEEP_MARGIN = 10.0   # km: "deep" intra-arc must still be ABOVE slab by at most this

# Subduction-domain zoning
SUBDUCTION_CLASSIFY_MAX_SLAB_DEPTH = 70.0   # km (only here classify interface/intraslab/forearc)
INTERFACE_DEPTH_TOL = 10.0                  # km window around slab for "interface"
FOREARC_MAX_DEPTH   = 60.0                  # km
SLAB_ABOVE_MARGIN   = 5.0                   # km (forearc must be at least this ABOVE slab)

# Mechanism matching for INTERFACE
CONV_AZIMUTH_DEG     = 78.0   # N78E, fixed convergence trend (degrees from North, clockwise)
CONV_AZ_TOL          = 45.0   # deg for slip azimuth vs convergence
USE_THRUST_RAKE_PREF = True
RAKE_THRUST_TARGET   = 90.0   # deg (thrust)
RAKE_TOL             = 35.0   # deg
USE_SLAB_DIP_MATCH   = True
DIP_TOL              = 20.0   # deg
USE_SLAB_STRIKE_MATCH= True
STRIKE_TOL           = 25.0   # deg (180° periodic)
USE_P_AXIS_CHECK     = True
P_AXIS_AZ_TOL        = 40.0   # deg |P_az - conv|
P_AXIS_MAX_PLUNGE    = 50.0   # deg

# Slab query cutoff
SLAB_QUERY_MAXDIST_KM = 15.0                # nearest slab node must be within this distance

# ------------------ INPUT PATHS ------------------
CATALOG_CSV     = "merged_all_catalogs.csv"
INTRA_ARC_SHP   = "../polygons/intraarc_polygon.shp"
TRENCH_LINE_SHP = "../shapefiles/chile_trench.shp"

SLAB_DEPTH_XYZ  = "../slab2/sam_slab2_dep_02.23.18.xyz"  # lon(0-360),lat,depth(km +down, NaN where undefined)
SLAB_STRIKE_XYZ = "../slab2/sam_slab2_str_02.23.18.xyz"  # lon(0-360),lat,strike(0-360 or NaN)
SLAB_DIP_XYZ    = "../slab2/sam_slab2_dip_02.23.18.xyz"  # lon(0-360),lat,dip(0-90 or NaN)

# ------------------ OUTPUT ------------------
OUT_DIR        = "./"
WRITE_COMBINED = True
COMBINED_CSV   = "catalog_labeled_v2.csv"

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

def angdiff_dir(a: float, b: float) -> float:
    """
    Directional difference between two azimuths in degrees (0..180).
    E.g., 350 vs 10 -> 20; 90 vs 270 -> 180.
    """
    d = abs((a - b) % 360.0)
    return d if d <= 180.0 else 360.0 - d

def angdiff_line(a: float, b: float) -> float:
    """
    **Line** (axis) difference, ignoring direction (0..90).
    90 and 270 are identical lines -> diff = 0.
    Works by folding modulo 180 and measuring the shortest angle to 0 or 180.
    """
    # map into [-90, +90] around b
    return abs(((a - b + 90.0) % 180.0) - 90.0)
def strike_diff(a, b):
    """Strike difference with 180° periodicity."""
    d = abs((a - b) % 180.0)
    return d if d <= 90.0 else 180.0 - d

def dip_diff(a, b):
    return abs(a - b)

# ------------------ slip azimuth from (strike,dip,rake) ------------------
def slip_azimuth_from_sdr(strike_deg, dip_deg, rake_deg) -> Optional[float]:
    """
    Returns azimuth (° from North, CW) of the slip vector's horizontal projection.
    """
    if any(pd.isna(v) for v in [strike_deg, dip_deg, rake_deg]):
        return None
    strike = math.radians(float(strike_deg))
    dip    = math.radians(float(dip_deg))
    rake   = math.radians(float(rake_deg))

    # Unit strike (N,E,Down; strike clockwise from North)
    u_s = np.array([math.cos(strike), math.sin(strike), 0.0])
    # Unit down-dip direction in plane
    u_d = np.array([-math.cos(dip)*math.sin(strike), math.cos(dip)*math.cos(strike), -math.sin(dip)])

    # Slip vector
    s_vec = math.cos(rake) * u_s + math.sin(rake) * u_d
    sN, sE = float(s_vec[0]), float(s_vec[1])
    if abs(sN) < 1e-12 and abs(sE) < 1e-12:
        return None
    az = (math.degrees(math.atan2(sE, sN)) + 360.0) % 360.0
    return az

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
        # convert lon from 0-360 to -180..180
        arr["lon"] = arr["lon"].astype(float).apply(lon_0360_to_180)
        arr["lat"] = arr["lat"].astype(float)
        return arr

    dep = read_xyz(depth_path)
    dep['val'] = -dep['val']
    st  = read_xyz(strike_path)
    di  = read_xyz(dip_path)

    # Join on lon-lat; assume same grid. If not exact, merge safely.
    merged = dep.merge(st, on=["lon","lat"], how="outer", suffixes=("_dep","_st"))
    merged = merged.merge(di, on=["lon","lat"], how="outer")
    merged.rename(columns={"val":"val_dip"}, inplace=True)

    # Drop rows where all are NaN
    merged = merged.dropna(subset=["val_dep","val_st","val_dip"], how="all")

    lon = merged["lon"].to_numpy(float)
    lat = merged["lat"].to_numpy(float)
    depth = merged["val_dep"].to_numpy(float)
    strike = merged["val_st"].to_numpy(float)
    dip = merged["val_dip"].to_numpy(float)

    # KDTree in degrees; we'll post-check by haversine distance
    coords = np.c_[lon, lat]
    tree = cKDTree(coords)
    return SlabGrid(tree=tree, lon=lon, lat=lat, depth=depth, strike=strike, dip=dip)

def query_slab(grid: SlabGrid, lon: float, lat: float, maxdist_km: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if np.isnan(lon) or np.isnan(lat):
        return (None, None, None)
    # Query nearest in degrees, then check real distance
    dist, idx = grid.tree.query(np.array([lon, lat]), k=1)
    if np.isinf(dist) or idx is None:
        return (None, None, None)
    dkm = haversine_km(lon, lat, grid.lon[idx], grid.lat[idx])
    if dkm > maxdist_km:
        return (None, None, None)
    # Get values (may be NaN)
    dep = grid.depth[idx]
    st  = grid.strike[idx]
    di  = grid.dip[idx]
    # Treat NaN as missing
    dep_val = None if pd.isna(dep) else float(dep)
    st_val  = None if pd.isna(st)  else float(st % 360.0)
    di_val  = None if pd.isna(di)  else float(di)
    return (dep_val, st_val, di_val)

# ------------------ geometry helpers ------------------
def load_intra_arc_polygon(path: str):
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs(epsg=4326)
    poly = gdf.unary_union
    return poly

def load_trench_line(path: str):
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs(epsg=4326)
    line = gdf.unary_union
    return line

def is_west_of_trench(lon: float, lat: float, trench_line) -> bool:
    """
    Simple heuristic: compare lon to nearest point on trench line.
    Works well for Chile margin (west is more negative longitude).
    """
    if trench_line is None or np.isnan(lon) or np.isnan(lat):
        return False
    pt = Point(lon, lat)
    nearest = trench_line.interpolate(trench_line.project(pt))
    return float(lon) < float(nearest.x)  # west if smaller longitude

# ------------------ mechanism checks for interface ------------------
def rake_close_to_thrust(rake: float, target: float = RAKE_THRUST_TARGET, tol: float = RAKE_TOL) -> bool:
    if rake is None or pd.isna(rake):
        return False
    # rake is typically in [-180,180]; measure |r - 90|
    r = float(rake)
    return abs(((r - target + 180.0) % 360.0) - 180.0) <= tol

def nodal_plane_matches_interface(strike: float, dip: float, rake: float,
                                  slab_strike: Optional[float], slab_dip: Optional[float],
                                  p_azimuth: Optional[float]) -> bool:
    """
    Returns True if this plane looks like an interface slip:
    - slip azimuth ~ fixed convergence azimuth (78°)
    - optional thrust rake preference
    - optional slab dip/strike compatibility
    - optional P-axis azimuth and plunge condition (we only have azimuth here)
    """
    # slip azimuth test
    # 1) Slip azimuth from SDR, compare to convergence trend as a LINE.
    saz = slip_azimuth_from_sdr(strike, dip, rake)  # your existing routine
    # if saz is None:
    #     return False
    # if angdiff_line(saz, CONV_AZIMUTH_DEG) > CONV_AZ_TOL:
    #     return False

    # 2) (Optional) slab strike/dip compatibility — also as a LINE for strike.
    # if (slab_strike is not None) and np.isfinite(slab_strike):
    #     if angdiff_line(float(strike), float(slab_strike)) > STRIKE_TOL:
    #         return False
    # if (slab_dip is not None) and np.isfinite(slab_dip):
    #     if abs(float(dip) - float(slab_dip)) > DIP_TOL:
    #         return False

    # 3) (Optional) P-axis azimuth close to convergence (as a LINE; P-axis has 180° ambiguity).
    # if (p_azimuth is not None) and np.isfinite(p_azimuth):
    #     if angdiff_line(float(p_azimuth), CONV_AZIMUTH_DEG) > P_AXIS_AZ_TOL:
    #         return False

    # 4) (Optional) thrust preference — keep loose; many centroid solutions wobble.
    # if USE_THRUST_RAKE_PREF and not rake_close_to_thrust(rake):
    #     return False

    return True


def is_interface_mechanism(row, slab_strike: Optional[float], slab_dip: Optional[float]) -> bool:
    """
    Consider both nodal planes; if either matches interface tests, return True.
    """
    s1, d1, r1 = row.get("strike1"), row.get("dip1"), row.get("rake1")
    s2, d2, r2 = row.get("strike2"), row.get("dip2"), row.get("rake2")
    p_az = row.get("P_azimuth")
    ok1 = (s1 is not None and d1 is not None and r1 is not None) and \
          nodal_plane_matches_interface(s1, d1, r1, slab_strike, slab_dip, p_az)
    if ok1:
        return True
    ok2 = (s2 is not None and d2 is not None and r2 is not None) and \
          nodal_plane_matches_interface(s2, d2, r2, slab_strike, slab_dip, p_az)
    return bool(ok2)

# ------------------ classification ------------------
def classify_row(row, intra_poly, trench_line, slab: SlabGrid) -> str:
    lon = float(row.get("longitude")) if row.get("longitude") is not None else np.nan
    lat = float(row.get("latitude")) if row.get("latitude") is not None else np.nan
    dep = float(row.get("depth")) if row.get("depth") is not None else np.nan

    # Slab query
    slab_depth, slab_strike, slab_dip = query_slab(slab, lon, lat, SLAB_QUERY_MAXDIST_KM)


    # Flags
    in_intra = False
    if intra_poly is not None and not (np.isnan(lon) or np.isnan(lat)):
        in_intra = intra_poly.contains(Point(lon, lat))

    west_of_tr = is_west_of_trench(lon, lat, trench_line)

    # 1) Outer-rise (oceanward of trench)
    if west_of_tr:
        return "outer_rise"

    # 2) Intra-arc shallow
    if in_intra and not np.isnan(dep) and dep <= INTRA_ARC_SHALLOW_MAX:
        return "crustal_intraarc_shallow"


    # 3) If slab is defined nearby
    if slab_depth is not None and not np.isnan(dep):
        # classify only in 'shallow slab' domain first
        if slab_depth <= SUBDUCTION_CLASSIFY_MAX_SLAB_DEPTH:
            depth_diff = abs(dep - slab_depth)
            # interface: near slab and mechanism consistent
            if depth_diff <= INTERFACE_DEPTH_TOL:# and is_interface_mechanism(row, slab_strike, slab_dip):
                return "subduction_interface"

            # intraslab: clearly below the interface
            if dep >= slab_depth + INTERFACE_DEPTH_TOL:
                return "subduction_intraslab"

            # forearc: above slab by margin, shallow enough, and not inside intra-arc polygon
            if (dep <= slab_depth - SLAB_ABOVE_MARGIN) and (dep <= FOREARC_MAX_DEPTH) and (not in_intra):
                return "forearc"

            # intra-arc deep (still above slab, but deeper than shallow threshold)
            if in_intra and (dep > INTRA_ARC_SHALLOW_MAX) and (dep <= slab_depth + INTRA_ARC_DEEP_MARGIN):
                return "crustal_intraarc_deep"

            # intra-arc but beneath slab
            if in_intra and (dep >= slab_depth + INTERFACE_DEPTH_TOL):
                return "deep"

        else:
            # slab deeper than 70 km ⇒ deep subduction domain
            # Near or below slab -> deep_subduction
            if (abs(dep - slab_depth) <= INTERFACE_DEPTH_TOL) or (dep >= slab_depth - INTERFACE_DEPTH_TOL):
                return "deep_subduction"

            # Above slab but deep could still fall here; keep unclassified for now
            # (intra-arc categories already handled above)
            pass

    # 4) Intra-arc deep (no slab nearby, or outside slab classify zone, but inside polygon)
    if in_intra and not np.isnan(dep) and dep > INTRA_ARC_SHALLOW_MAX:
        return "crustal_intraarc_deep"

    # 5) Forearc shallow (no slab info but shallow and not intra-arc, not outer-rise)
    if not in_intra and not np.isnan(dep) and dep <= FOREARC_MAX_DEPTH:
        return "forearc"

    # 6) Fallback
    return "unclassified"

# ------------------ IO + main ------------------
if __name__ == "__main__":

    # Load inputs
    df = pd.read_csv(CATALOG_CSV)
    # normalize numeric
    for col in ["longitude","latitude","depth","strike1","dip1","rake1","strike2","dip2","rake2",
                "T_plunge","T_azimuth","N_plunge","N_azimuth","P_plunge","P_azimuth",
                "Mrr","Mtt","Mpp","Mrt","Mrp","Mtp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Load spatial data
    intra_poly = load_intra_arc_polygon(INTRA_ARC_SHP) if os.path.exists(INTRA_ARC_SHP) else None
    trench_line = load_trench_line(TRENCH_LINE_SHP) if os.path.exists(TRENCH_LINE_SHP) else None

    # Load slab grids
    slab = load_slab_xyz(SLAB_DEPTH_XYZ, SLAB_STRIKE_XYZ, SLAB_DIP_XYZ)

    # Classify
    classes = []
    for _, row in df.iterrows():
        cls = classify_row(row, intra_poly, trench_line, slab)
        classes.append(cls)
    df["class"] = classes

    # Write combined
    if WRITE_COMBINED:
        df.to_csv(COMBINED_CSV, index=False)
        print(f"Wrote {COMBINED_CSV} with {len(df)} rows.")

    # Write per-class
    os.makedirs(OUT_DIR, exist_ok=True)
    for cls in CLASSES:
        sub = df[df["class"] == cls].copy()
        outp = os.path.join(OUT_DIR, f"{cls}.csv")
        sub.to_csv(outp, index=False)
        print(f"Wrote {outp} ({len(sub)} rows).")

