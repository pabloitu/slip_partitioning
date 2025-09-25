import math
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS

# --- your helpers (kept as-is) ---
def lon_0360_to_180(lon):
    lon = float(lon)
    return lon - 360.0 if lon > 180.0 else lon

def _read_xyz(p):
    arr = pd.read_csv(p, header=None, names=["lon","lat","val"])
    arr["lon"] = arr["lon"].astype(float).apply(lon_0360_to_180)
    arr["lat"] = arr["lat"].astype(float)
    # keep NaNs — they represent missing cells we want to preserve in the raster
    return arr

def xyz_to_geotiff(xyz_path: str, out_tif: str, dtype="float32"):
    """Create a north-up GeoTIFF from an XYZ file where (lon,lat) are cell centers."""
    df = _read_xyz(xyz_path)

    # Sort unique coordinates; longitudes west→east, latitudes south→north initially
    lons = np.sort(df["lon"].unique())
    lats = np.sort(df["lat"].unique())

    if len(lons) < 2 or len(lats) < 2:
        raise ValueError("Need at least a 2x2 grid of lon/lat to define raster resolution.")

    # Grid spacing (assume quasi-regular): use robust step via median diff
    dx = float(np.median(np.diff(lons)))
    dy = float(np.median(np.diff(lats)))
    if dx <= 0 or dy <= 0:
        raise ValueError("Non-positive grid spacing detected. Check coordinate values.")

    # Pivot to 2D: pivot index=lat (rows), columns=lon (cols), values=val
    # This yields rows ordered south→north; for north-up rasters we want north→south
    grid = df.pivot(index="lat", columns="lon", values="val").reindex(index=lats, columns=lons)

    # Ensure array is (rows: north→south, cols: west→east)
    data = np.flipud(grid.to_numpy())  # flip rows so top = largest lat (north)

    # Compute geotransform from cell centers:
    # west edge is min_lon_center - dx/2, north edge is max_lat_center + dy/2
    west_edge  = lons.min() - dx / 2.0
    north_edge = lats.max() + dy / 2.0
    transform = Affine.translation(west_edge, north_edge) * Affine.scale(dx, -dy)

    height, width = data.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": dtype,
        "crs": CRS.from_epsg(4326),
        "transform": transform,
        "tiled": True,
        "compress": "deflate",
        "nodata": np.nan,
    }

    # Cast to desired dtype; keep NaNs for float types
    data_to_write = data.astype(dtype)

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(data_to_write, 1)


xyz_to_geotiff("../slab2/sam_slab2_dep_02.23.18.xyz", "depth.tif")
xyz_to_geotiff("../slab2/sam_slab2_str_02.23.18.xyz", "strike.tif")
xyz_to_geotiff("../slab2/sam_slab2_dip_02.23.18.xyz", "dip.tif")
