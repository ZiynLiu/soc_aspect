# soc_aspect

This repository contains workflows for mountain soil organic carbon analysis using Google Earth Engine, topographic stratification, and AEI calculation.

## File Functions

- `code/1_calculate_mountain_area.ipynb`: Identifies and classifies mountain areas from ALOS DEM data using elevation, slope, and local elevation range rules.
- `code/2_modeling_for_socd_in_gee.ipynb`: Trains and tunes a Random Forest model in Google Earth Engine to predict SOCD from satellite embedding features and soil observations.
- `code/2b_downloading_tiiles_for_spatial_comparison.ipynb`: Applies the trained SOCD model to generate prediction tiles for spatial comparison and export-oriented analysis.
- `code/3_statistic_topography_groups.py`: Groups SOC pixels by elevation, slope, and aspect classes within fishnet grids and exports grouped zonal statistics in batches.
- `code/4_calculating_AEI.ipynb`: Parses grouped statistics, links them with biome and grid data, and computes the AEI metric from topographic response patterns.
- `code/download_dem.ipynb`: Downloads elevation, slope, and aspect rasters for selected fishnet grids and visualizes SOCD prediction layers for local inspection.
