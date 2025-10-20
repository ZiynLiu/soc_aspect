import ee
import geemap
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
import time

# Earth Engine Authentication and Initialization
ee.Authenticate()
ee.Initialize(project='socd-liuziyan')

# ==================== Parameter Configuration (Refactored to Median Composite Mode) ====================
project_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

params = {
    'scale': 10,
    'tile_scale': 16,
    'elev_bin_size': 100,
    'slope_bin_size': 5,
    'aspect_bin_size': 10,
    'batch_size': 100,
    'output_dir': r'D:\GEE_EVI_aspect_effect_30m_20250922',
    'start_from_point': 0,
    'retry_delay': 10,
    'max_wait_time': 300,
    'export_to_drive': True,
    'base_folder_name': f'EVI_Median_1984_2024_{project_timestamp}',
    'test_mode': False,
    'composite_start_year': 1984,
    'composite_end_year': 2024,
    'composite_method': 'median',
}

# ==================== Load EVI Collection and Composite Median for Growing Season (Optimized: Annual -> Inter-annual Composite) ====================
# Load EVI 8-day image collection and filter for 1984-2024 (extending to April of the next year for Southern Hemisphere)
evi_all = ee.ImageCollection("LANDSAT/COMPOSITES/C02/T1_L2_8DAY_EVI") \
    .select('EVI') \
    .filterDate(f"{params['composite_start_year']}-01-01", f"{params['composite_end_year'] + 1}-04-01") \
    .map(lambda img: img.updateMask(img.gte(0)))

# Filter growing season by hemisphere and calculate annual median, then calculate median of annual images
years = ee.List.sequence(params['composite_start_year'], params['composite_end_year'])

# Hemisphere mask (latitude > 0 is Northern Hemisphere)
north_mask = ee.Image.pixelLonLat().select('latitude').gt(0)

def nh_annual_median(y):
    y = ee.Number(y)
    # June to September of the current year
    season_ic = evi_all \
        .filter(ee.Filter.calendarRange(6, 9, 'month')) \
        .filter(ee.Filter.calendarRange(y, y, 'year'))
    return season_ic.median().updateMask(north_mask).set('year', y)

def sh_annual_median(y):
    y = ee.Number(y)
    # December of the current year + January to March of the next year (cross-year)
    dec_ic = evi_all \
        .filter(ee.Filter.calendarRange(12, 12, 'month')) \
        .filter(ee.Filter.calendarRange(y, y, 'year'))
    jan_mar_ic = evi_all \
        .filter(ee.Filter.calendarRange(1, 3, 'month')) \
        .filter(ee.Filter.calendarRange(y.add(1), y.add(1), 'year'))
    season_ic = ee.ImageCollection(dec_ic.merge(jan_mar_ic))
    return season_ic.median().updateMask(north_mask.Not()).set('year', y)

nh_annual_ic = ee.ImageCollection(years.map(nh_annual_median))
sh_annual_ic = ee.ImageCollection(years.map(sh_annual_median))

# Calculate the median of the "annual seasonal median image collections" to get the long-term growing season EVI
nh_median_evi = nh_annual_ic.median().rename('evi')
sh_median_evi = sh_annual_ic.median().rename('evi')

# Combine global: complementary masks for the two hemispheres, simply blend
evi_composite = nh_median_evi.blend(sh_median_evi).rename('evi')

# ==================== Mountain Mask ====================
# Load UNEP mountain classification and generate mask
mountain_classification = ee.Image("projects/socd-liuziyan/assets/global_mountain_classification_UNEP")
mountain_mask = mountain_classification.gt(0)
mountain_areas = mountain_classification.updateMask(mountain_mask).gt(0).toInt8()
kernel = ee.Kernel.circle(radius=2500, units='meters', normalize=True)
dilated = mountain_areas.focal_max(kernel=kernel, iterations=1)
closed_mask = dilated.focal_min(kernel=kernel, iterations=1)

evi_composite_masked = evi_composite.updateMask(closed_mask)

# ==================== Topographic Data and Binning ====================
# Load and process topographic data (ALOS AW3D30)
dataset = ee.ImageCollection('JAXA/ALOS/AW3D30/V4_1')
elevation_collection = dataset.select('DSM')
proj = elevation_collection.first().select(0).projection()
elevation = elevation_collection.mosaic().setDefaultProjection(proj)

slope = ee.Terrain.slope(elevation)
aspect = ee.Terrain.aspect(elevation)

scale = params['scale']
elevation_resampled = elevation.resample('bicubic').reproject(crs='EPSG:4326', scale=scale)
slope_resampled = slope.resample('bicubic').reproject(crs='EPSG:4326', scale=scale)
aspect_resampled = aspect.resample('bicubic').reproject(crs='EPSG:4326', scale=scale)

# Binning
elev_bin = elevation_resampled.divide(params['elev_bin_size']).floor().int().rename('elev_bin')

# Create custom slope bins
slope_bin = ee.Image.constant(6)
slope_bin = slope_bin.where(slope_resampled.lt(55), 5)
slope_bin = slope_bin.where(slope_resampled.lt(45), 4)
slope_bin = slope_bin.where(slope_resampled.lt(35), 3)
slope_bin = slope_bin.where(slope_resampled.lt(25), 2)
slope_bin = slope_bin.where(slope_resampled.lt(15), 1)
slope_bin = slope_bin.where(slope_resampled.lt(5), 0)
slope_bin = slope_bin.int().rename('slope_bin')

# Create custom aspect bins
distance_from_south = aspect_resampled.subtract(180).abs()
aspect_bin = distance_from_south.divide(10).floor().int().rename('aspect_bin')

# Create composite grouping key
composite_key = elev_bin.multiply(1000000).add(
    slope_bin.multiply(1000)
).add(aspect_bin).rename('composite_key')

# ==================== Fishnet Grid ====================
fishnet = ee.FeatureCollection("projects/socd-liuziyan/assets/fishnet_grid_mountain_50km_with_id")
grid_count = fishnet.size().getInfo()

# ==================== Reducer ====================
def create_optimized_zonal_reducer():
    return ee.Reducer.mean().combine(
        ee.Reducer.count(),
        sharedInputs=True
    ).group(
        groupField=1,
        groupName='composite_key'
    )

# ==================== Batch Processing Function (Based on Composite Image) ====================
def process_batches(evi_img_masked, export_folder, composite_label):
    analysis_image = ee.Image.cat([
        evi_img_masked,  # Band 0
        composite_key    # Band 1
    ])
    fishnet_list = fishnet.toList(fishnet.size())
    total = grid_count
    start_from_point = params['start_from_point']
    remaining_points = total - start_from_point
    total_batches = (remaining_points + params['batch_size'] - 1) // params['batch_size']
    start_batch_idx = start_from_point // params['batch_size']

    if params['test_mode']:
        batch_range = range(start_batch_idx, start_batch_idx + 1)
    else:
        batch_range = range(start_batch_idx, (total + params['batch_size'] - 1) // params['batch_size'])

    export_tasks = []
    global_start = time.time()

    for batch_idx in batch_range:
        start = batch_idx * params['batch_size']
        end = min(start + params['batch_size'], total)
        if end <= start_from_point:
            continue
        if start < start_from_point:
            start = start_from_point
        actual_batch_size = end - start

        batch_start_time = time.time()
        try:
            batch_fc = ee.FeatureCollection(fishnet_list.slice(start, end))
            zonal_reducer = create_optimized_zonal_reducer()
            zonal_stats = analysis_image.reduceRegions(
                collection=batch_fc,
                reducer=zonal_reducer,
                scale=params['scale'],
                tileScale=params['tile_scale']
            )
            task_description = f"evi_{composite_label}_batch_{batch_idx:03d}_{start:05d}_{end:05d}"
            export_task = ee.batch.Export.table.toDrive(
                collection=zonal_stats,
                description=task_description,
                folder=export_folder,      # Unified single folder
                fileNamePrefix=task_description,
                fileFormat='CSV'
            )
            export_task.start()
            export_tasks.append({
                'task': export_task,
                'batch_idx': batch_idx,
                'description': task_description,
                'point_range': f'{start}-{end}',
                'start_time': batch_start_time
            })
            if params['test_mode']:
                break
        except Exception as e:
            if params['test_mode']:
                break

    return export_tasks

# ==================== Main Execution ====================
composite_label = (f"{params['composite_start_year']}_{params['composite_end_year']}_median"
                   if params['composite_method'] == 'median'
                   else f"{params['composite_start_year']}_{params['composite_end_year']}_mean")

# Use a single Drive folder
export_folder = params['base_folder_name']

tasks = process_batches(evi_composite_masked, export_folder, composite_label)