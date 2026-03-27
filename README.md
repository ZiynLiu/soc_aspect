# soc_aspect

## Code Folder Overview

The `code/` folder contains the main workflow notebooks and scripts for terrain-aware SOC analysis in Google Earth Engine (GEE) and downstream statistics.

- `1_calculate_mountain_area.ipynb`  
  Calculates mountain-area masks and related topographic layers (primarily based on ALOS AW3D30 DEM) to define the analysis extent.

- `2_modeling_for_socd_in_gee.ipynb`  
  Builds and tunes a Random Forest model in GEE to predict SOC density (SOCD) using SOC sample points and AlphaEarth satellite embedding features, then prepares prediction/export outputs.

- `2b_downloading_tiiles_for_spatial_comparison.ipynb`  
  Runs a tile-based SOCD modeling/prediction workflow (similar to Notebook 2) focused on downloading tiles for spatial comparison and validation.

- `3_statistic_topography_groups.py`  
  Performs batch zonal statistics in GEE by grouping pixels with a composite topography key (elevation bin + slope bin + aspect bin), then exports grouped SOC summaries to Google Drive as CSV files.

- `4_calculating_AEI.ipynb`  
  Processes exported grouped statistics and computes AEI-related metrics using Python data analysis/geospatial libraries (`pandas`, `geopandas`, `scipy`).

- `download_dem.ipynb`  
  Downloads/extracts DEM and related raster inputs for selected fishnet grids to support local inspection, model inputs, or spatial comparison tasks.
