import os
from pathlib import Path

def find_project_root(start=None) -> Path:
    p = Path(start or __file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()

# Directories
ROOT = Path(os.environ.get("PROJECT_ROOT", find_project_root()))
DATA = ROOT / "data"
CATALOG_DIR = DATA / "seismicity"
REGIONAL_CATALOG_DIR = CATALOG_DIR / "regional"
LOCAL_CATALOGS_DIR = CATALOG_DIR / "local"
SLAB_DIR = DATA / "slab_2.0"
SHAPEFILE_DIR = DATA / "shapefiles"

# Shapefiles
intraarc_shp = SHAPEFILE_DIR / "intra_arc.shp"
trench_shp = SHAPEFILE_DIR / "sam_nazca_trench.shp"

# Slab 2.0
slab_depth =  SLAB_DIR / "sam_slab2_dep_02.23.18.xyz"
slab_strike = SLAB_DIR / "sam_slab2_str_02.23.18.xyz"
slab_dip =  SLAB_DIR / "sam_slab2_dip_02.23.18.xyz"

# Raw Catalogs
rawcat_anss = REGIONAL_CATALOG_DIR / "ANSS.csv"
rawcat_isc = REGIONAL_CATALOG_DIR / "ISC_bulletin.txt"
rawcat_isc_gem = REGIONAL_CATALOG_DIR / "ISC_GEM.csv"
rawcat_gcmt_1976_2020 = REGIONAL_CATALOG_DIR / "gcmt_jan76_dec20.txt"
rawcat_gcmt_2020_2025 = REGIONAL_CATALOG_DIR / "gcmt_jan21_aug25.txt"
rawcat_stanton = REGIONAL_CATALOG_DIR / "stanton_subduction_eqs.csv"
raw_potin = REGIONAL_CATALOG_DIR / "CHILE_SEISMICITY_RELOCATED.csv"

# Output paths
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
CATALOGS = RESULTS / "catalogs"
FORMATTED_CATALOGS = CATALOGS / "formatted"
MERGED_CATALOGS = CATALOGS / "merge"
RELOCATED_CATALOGS = CATALOGS / "relocated"

CLASSIFIED_CATALOGS = CATALOGS / "classified"
BEACHBALLS = CATALOGS / "beachballs"

for d in (RESULTS, FIGURES, FORMATTED_CATALOGS, MERGED_CATALOGS, CLASSIFIED_CATALOGS, BEACHBALLS, RELOCATED_CATALOGS):
    d.mkdir(parents=True, exist_ok=True)

# Processed Catalogs
cat_anss = FORMATTED_CATALOGS / "anss.csv"
cat_gcmt_perez = FORMATTED_CATALOGS / "gcmt_perez.csv"
cat_gcmt = FORMATTED_CATALOGS / "gcmt.csv"
cat_isc = FORMATTED_CATALOGS / "isc.txt"
cat_isc_gem = FORMATTED_CATALOGS / "isc_gem.csv"
cat_potin = FORMATTED_CATALOGS / "potin.csv"

# Merge
cat_merged = MERGED_CATALOGS / "cat_merged.csv"
cat_full = MERGED_CATALOGS / "cat_full.csv"

# Relocated
cat_relocated = RELOCATED_CATALOGS / "cat_relocated.csv"

# Classified
anss_classified = CLASSIFIED_CATALOGS / "anss_classified.csv"
gcmt_classified = CLASSIFIED_CATALOGS / "gcmt_classified.csv"
merged_classified = CLASSIFIED_CATALOGS / "merged_classified.csv"
full_classified = CLASSIFIED_CATALOGS / "full_classified.csv"
relocated_classified = CLASSIFIED_CATALOGS / "relocated_classified.csv"
selected_classified = CLASSIFIED_CATALOGS / "selected_classified.csv"
