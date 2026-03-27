import ee
import geemap
from datetime import datetime
import math
import time

ee.Authenticate()
ee.Initialize(project="")  # change to your own project if needed

params = {
    "scale": 30,
    "tile_scale": 8,
    "elev_bin_size": 100,
    "aspect_bin_size": 10,
    "batch_size": 50,
    "start_batch": 0,
    "sleep_seconds": 2,
    "drive_folder": "SOC_GSM_GMBA_50km_grouped_batches",
    "export_description_prefix": "soc_gsm_grouped",
}

# change to your own SOC image if needed
soc_img = ee.Image("")  # change to your own SOC image if needed
soc_img = soc_img.select(["mg_cm3"], ["soc"])

dataset = ee.ImageCollection("JAXA/ALOS/AW3D30/V4_1")
elevation_collection = dataset.select("DSM")
proj = elevation_collection.first().select(0).projection()
elevation = elevation_collection.mosaic().setDefaultProjection(proj)

slope = ee.Terrain.slope(elevation)
aspect = ee.Terrain.aspect(elevation)

scale = params["scale"]
elevation_resampled = elevation.resample("bicubic").reproject(
    crs="EPSG:4326",
    scale=scale,
)
slope_resampled = slope.resample("bicubic").reproject(
    crs="EPSG:4326",
    scale=scale,
)
aspect_resampled = aspect.resample("bicubic").reproject(
    crs="EPSG:4326",
    scale=scale,
)

elev_bin_base = (
    elevation_resampled.divide(params["elev_bin_size"]).floor().int().rename("elev_bin")
)

slope_bin_base = ee.Image.constant(6)
slope_bin_base = slope_bin_base.where(slope_resampled.lt(55), 5)
slope_bin_base = slope_bin_base.where(slope_resampled.lt(45), 4)
slope_bin_base = slope_bin_base.where(slope_resampled.lt(35), 3)
slope_bin_base = slope_bin_base.where(slope_resampled.lt(25), 2)
slope_bin_base = slope_bin_base.where(slope_resampled.lt(15), 1)
slope_bin_base = slope_bin_base.where(slope_resampled.lt(5), 0)
slope_bin_base = slope_bin_base.int().rename("slope_bin")

distance_from_south = aspect_resampled.subtract(180).abs()
aspect_bin_base = (
    distance_from_south.divide(params["aspect_bin_size"])
    .floor()
    .int()
    .min(17)
    .rename("aspect_bin")
)

composite_key_base = (
    elev_bin_base.multiply(1000000)
    .add(slope_bin_base.multiply(1000))
    .add(aspect_bin_base)
    .rename("composite_key")
)

fishnet = ee.FeatureCollection("")  # change to your own fishnet feature collection if needed


def keep_id_only(feature):
    return ee.Feature(feature.geometry(), {"id": feature.get("id")})


fishnet = fishnet.map(keep_id_only)

grid_count = fishnet.size().getInfo()
total_batches = math.ceil(grid_count / params["batch_size"])
fishnet_list = fishnet.toList(grid_count)


def create_zonal_reducer():
    return ee.Reducer.mean().combine(
        ee.Reducer.count(),
        sharedInputs=True,
    ).group(
        groupField=1,
        groupName="composite_key",
    )


def build_analysis_image():
    soc_img_resampled = (
        soc_img.resample("bicubic")
        .reproject(crs="EPSG:4326", scale=params["scale"])
        .rename("soc")
    )

    soc_mask = soc_img_resampled.mask()
    composite_key = composite_key_base.updateMask(soc_mask)

    return ee.Image.cat([soc_img_resampled, composite_key])


analysis_image = build_analysis_image()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tasks = []

for batch_idx in range(params["start_batch"], total_batches):
    start = batch_idx * params["batch_size"]
    end = min(start + params["batch_size"], grid_count)

    batch_fc = ee.FeatureCollection(fishnet_list.slice(start, end))

    zonal_stats = analysis_image.reduceRegions(
        collection=batch_fc,
        reducer=create_zonal_reducer(),
        scale=params["scale"],
        tileScale=params["tile_scale"],
    )

    description = (
        f'{params["export_description_prefix"]}'
        f"_b{batch_idx:04d}_{start}_{end}_{timestamp}"
    )

    task = ee.batch.Export.table.toDrive(
        collection=zonal_stats,
        description=description,
        folder=params["drive_folder"],
        fileNamePrefix=description,
        fileFormat="CSV",
    )

    task.start()
    tasks.append(
        {
            "batch_idx": batch_idx,
            "start": start,
            "end": end,
            "description": description,
        }
    )

    time.sleep(params["sleep_seconds"])

for task_info in tasks:
    print(
        f'batch {task_info["batch_idx"]} | '
        f'{task_info["start"]}-{task_info["end"]} | '
        f'{task_info["description"]}'
    )