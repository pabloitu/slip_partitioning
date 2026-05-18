## Data Repository: 
### Exploring slip partitioning in the Southern Andes: New insights from fault slip data and crustal seismicity. 
#### Cembrano et al., submitted to Andean Geology special issue (2025)


### 1. Focal Mechanism Regional Compilation From Global Sources (`catalogs/regional_scale`)


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


### 2. Local seismic-network catalog (catalog/local_networks/)

Compilation of focal mechanism from local seismic network deployments directly from the listed references, some of which were digitized by Perez-Estay et al., (2023). The P and T axes, as well as the secondary nodal planes, were calculated using Python code based on the libraries `Beachball`, `mplstereonet` and `FMC`.


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


### 3. Fault-slip dataset (fault_data/)

Field data was obtained by the research group since 1990's to the present. In particular, fault slip data was recorded at different sites of the SVZ by different members identifying plane geometry (strike and dip), and clear kinematic indicators to identify rake and sense of slip. Only reliable data was included in this compilation, in which measurement and sense of slip were clearly identified from field observations. With the use of `Faultkin v8.0` software (Marret and Allmendinger, 1990; Allmendinger et al., 2012), P and T (maximum shortening and elongation axes) were obtained for each fault-slip datum individually.


| Column                       | Units / Type | Description                                                                                                                                                                                                                                                                                                                                                                                             |
| ---------------------------- | ------------ |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Source`                     | string       | This column specifies if the field data has already been published in other research publications from members of the group. In case the datum has not been published, it is labeled as "Unpublished Data".                                                                                                                                                                                             |
| `Structural site`            | string       | Field data was recorded associated to an area/site, indicated by the field geologist.                                                                                                                                                                                                                                                                                                                   |
| `Zone`                       | string       | UTM Zone associated to the coordinates (e.g., `18S`).                                                                                                                                                                                                                                                                                                                                                   |
| `UTM N`                      | m            | UTM northing.                                                                                                                                                                                                                                                                                                                                                                                           |
| `UTM E`                      | m            | UTM easting.                                                                                                                                                                                                                                                                                                                                                                                            |
| `Maximum age of deformation` | string       | Age of the rock in which the structural data was measured, associated with the maximum age of deformation. In case the datum has already been published, this information was taken from these publications. For unpublished data, rock age was obtained from: Munizaga et al., 1988; Carrasco 1995; Guzmán-Marín 2015; Peña et al., 2021; Piquer et al., 2017. All references sited in the manuscript. |
| `Fault strike`               | deg          | Fault plane strike in right-hand rule, 0 to 360 (clockwise from North).                                                                                                                                                                                                                                                                                                                                 |
| `Fault dip`                  | deg          | Fault plane dip in right-hand rule, 0 to 90 (from horizontal).                                                                                                                                                                                                                                                                                                                                          |
| `Rake`                       | deg          | Slip rake in right-hand rule, 0 to 180 (in plane, from strike toward dip).                                                                                                                                                                                                                                                                                                                              |
| `Sense of Slip`              | code         | Sense of slip inferred in the field from kinematic indicators (`NL`: normal/left-lateral, `NR`:normal/right-lateral, `TL`: thrust/left-lateral, `TR`: thrust/right-lateral).                                                                                                                                                                                                                            |
| `P`                          | deg/deg      | Attitude of the P axis (maximum shortening), in AA/BBB format, where AA corresponds to the plunge and BBB to the trend.                                                                                                                                                                                                                                                                                 |
| `T`                          | deg/deg      | Attitude of the T axis (maximum elongation), in AA/BBB format, where AA corresponds to the plunge and BBB to the trend.                                                                                                                                                                                                                                                                                 |
