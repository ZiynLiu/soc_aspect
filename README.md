# soc_aspect

<<<<<<< HEAD
## Code Overview

This repository contains workflows for mountain soil organic carbon analysis using Google Earth Engine, topographic stratification, and AEI calculation.

## File Functions

- `code/1_calculate_mountain_area.ipynb`: Identifies and classifies mountain areas from ALOS DEM data using elevation, slope, and local elevation range rules.
- `code/2_modeling_for_socd_in_gee.ipynb`: Trains and tunes a Random Forest model in Google Earth Engine to predict SOCD from satellite embedding features and soil observations.
- `code/2b_downloading_tiiles_for_spatial_comparison.ipynb`: Applies the trained SOCD model to generate prediction tiles for spatial comparison and export-oriented analysis.
- `code/3_statistic_topography_groups.py`: Groups SOC pixels by elevation, slope, and aspect classes within fishnet grids and exports grouped zonal statistics in batches.
- `code/4_calculating_AEI.ipynb`: Parses grouped statistics, links them with biome and grid data, and computes the AEI metric from topographic response patterns.
- `code/download_dem.ipynb`: Downloads elevation, slope, and aspect rasters for selected fishnet grids and visualizes SOCD prediction layers for local inspection.
=======
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
>>>>>>> 68d985ba6ef5653a7c83794b09b5b640e6db1df0
