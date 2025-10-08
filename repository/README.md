# Data Repository: Exploring slip partitioning in the Southern Andes: New insights from fault slip data and crustal seismicity. Cembrano et al., submitted to Andean Geology special issue (2015)


This repository provides all datasets used in our study of short- and long-term deformation across the Andean margin, focusing on crustal domains (forearc, intra-arc, and back-arc). It contains three main components:

## 1. Focal Mechanism Regional Compilation From Global Sources (`catalogs/regional_scale`)


This dataset merges focal mechanism and moment tensor solutions from the GCMT and ANSS/ComCat global catalogs (1976–2025), filtered for the study region and classified into tectonic domains. Several processing stages are provided:

   * `raw/` – Original downloaded catalogs.

   * `formatted/` – Standardized versions with consistent headers and units.

   * `processed/` – Final catalogs
     * `1_catalog_merged.csv`: GCMT (Erkstrom et al., 2012) and ANSS/ComCat (U.S. Geological Survey, 2017) merged and duplicates removed.
     * `2_catalog_merged_relocated.csv`: Catalog with improved hypocenters from relocated solutions (Potin et al., 2025), if available.
     * `3a_complete_catalog_classified_supmat.csv`: Classification for the complete catalog (Figure S1, Supplementary Material)
   * `3b_selection_catalog_classified_supmat.csv`: Classification for a selected catalog, including only crustal events plus a selection of subduction interface events (Figure 2, Manuscript)

The catalog data format is as follows:

| Column        | Units / Type    | Description                                                                                                                                                  |
| ------------- |-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `id`          | string          | Event identifier (native to source).                                                                                                                         |
| `time_iso`    | ISO-8601 UTC    | Origin time.                                                                                                                                                 |
| `longitude`   | deg             | Epicenter longitude (WGS84).                                                                                                                                 |
| `latitude`    | deg             | Epicenter latitude (WGS84).                                                                                                                                  |
| `depth`       | km              | Hypocentral depth (positive downwards).                                                                                                                      |
| `mag`         | numeric         | Reported magnitude (Mw preferred).                                                                                                                           |
| `mag_type`    | string          | Magnitude type (`mww`, `mwc`, `mwb`, `ml`).                                                                                                                  |
| `lon_error`   | km              | Longitude uncertainty (blank when not reported and “relocated” when hypocentral data was obtained from Potin et al., 2025).                                  |
| `lat_error`   | km              | Latitude uncertainty (blank when not reported and “relocated” when hypocentral data was obtained from Potin et al., 2025).                                   |
| `depth_error` | km              | Depth uncertainty (blank when not reported and “relocated” when hypocentral data was obtained from Potin et al., 2025).                                      |
| `mag_error`   | magnitude units | Magnitude uncertainty (if provided).                                                                                                                         |
| `strike1`     | deg             | Nodal plane 1 strike, 0 to 360 (clockwise from North).                                                                                                       |
| `dip1`        | deg             | Nodal plane 1 dip, 0 to 90 (from horizontal).                                                                                                                |
| `rake1`       | deg             | Nodal plane 1 rake, -180 to 180 (in plane, from strike toward dip).                                                                                          |
| `strike2`     | deg             | Nodal plane 2 strike.                                                                                                                                        |
| `dip2`        | deg             | Nodal plane 2 dip.                                                                                                                                           |
| `rake2`       | deg             | Nodal plane 2 rake.                                                                                                                                          |
| `Mrr`         | N·m             | Moment tensor rr component (GCMT convention).                                                                                                                |
| `Mtt`         | N·m             | Moment tensor tt component.                                                                                                                                  |
| `Mpp`         | N·m             | Moment tensor pp component.                                                                                                                                  |
| `Mrt`         | N·m             | Moment tensor rt component.                                                                                                                                  |
| `Mrp`         | N·m             | Moment tensor rp component.                                                                                                                                  |
| `Mtp`         | N·m             | Moment tensor tp component.                                                                                                                                  |
| `source`      | string          | Catalog of origin (`gcmt`, `anss`, …).                                                                                                                       |
| `dups`        | string          | Semicolon-separated IDs of de-duplicated cluster mates.                                                                                                      |
| `class`       | string          | Tectonic class (e.g., `forearc`, `intraarc_shallow`, `intraarc_deep`, `backarc`, `slab_interface`, `intra_slab`, `slab_deep`, `outer_rise`, `unclassified`). |
| `sub_depth`   | km              | Slab2 interface depth beneath epicenter (blank if N/A).                                                                                                      |

## 2. Local seismic-network catalog (catalog/local_networks/)

High-density arrays (typically 1–3 year windows) provide focal mechanisms for small events (often Mw < 4.5).

| Column                   | Units / Type | Description                                         |
|--------------------------|--------------|-----------------------------------------------------|
| `Source`                 | string       | Study/network reference (e.g., “Pérez-Estay 2020”). |
| `ID`                     | string/int   | ID used in this work.                               |
| `Source ID`              | string/int   | Event ID used by the source.                        |
| `Date`                   | YYYY/MM/DD   | Origin date.                                        |
| `HH:MM`                  | time         | Origin time (local or UTC per source).              |
| `Lat`                    | deg          | Epicenter latitude (WGS84).                         |
| `Lon`                    | deg          | Epicenter longitude (WGS84).                        |
| `Depth`                  | km           | Hypocentral depth (positive downwards).             |
| `Mw/Ml`                  | float/None   | Reported magnitude and type, if given.              |
| `n° Obs.`                | int          | Number of observations/stations (if reported).      |
| `Strike (Nodal Plane 1)` | deg          | Nodal plane 1 strike (0 to 360).                    |
| `Dip (Nodal Plane 1)`    | deg          | Nodal plane 1 dip (0 to 90).                        |
| `Rake (Nodal Plane 1)`   | deg          | Nodal plane 1 rake (-180 to 180).                   |
| `P trend`                | deg          | P-axis trend.                                       |
| `P plunge`               | deg          | P-axis plunge.                                      |
| `T trend`                | deg          | T-axis trend.                                       |
| `T plunge`               | deg          | T-axis plunge.                                      |
| `Strike (Nodal Plane 2)` | deg          | Nodal plane 2 strike (0 to 360).                    |
| `Dip (Nodal Plane 2)`    | deg          | Nodal plane 2 dip (0 to 90).                        |
| `Rake(Nodal Plane 2)`    | deg          | Nodal plane 2 rake (-180 to 180).                   |
| `Legend`                 | string       | Color Notes/provenance from source.                 |


## 3. Fault-slip dataset (fault_data/)

Mesoscopic measurements documenting fault slip data (strike–dip–rake, kinematics, and derived P–T axes) at mapped structural sites.

| Column                       | Units / Type | Description                                               |
| ---------------------------- | ------------ |-----------------------------------------------------------|
| `Source`                     | string       | Data origin (publication or “Unpublished Data”).          |
| `Structural site`            | string       | Site name.                                                |
| `Zone`                       | string       | UTM zone (e.g., `18S`).                                   |
| `UTM N`                      | m            | UTM northing.                                             |
| `UTM E`                      | m            | UTM easting.                                              |
| `Maximum age of deformation` | string       | Stratigraphic/chronologic ceiling (e.g., “114 Ma”).       |
| `Fault strike`               | deg          | Fault plane strike, 0 to 360 (clockwise from North).      |
| `Fault dip`                  | deg          | Fault plane dip, 0 to 90 (from horizontal).               |
| `Rake`                       | deg          | Slip rake, 0 to 180 (in plane, from strike toward dip).   |
| `Sense of Slip`              | code         | Kinematic label from source (e.g., `SS`, `N`, `R`, `TR`). |
| `P`                          | deg/deg      | Shortening axis as `plunge/trend`.                        |
| `T`                          | deg/deg      | Extension axis as `trend/plunge`.                         |
