import ee
import pandas as pd
import os
import numpy as np

ee.Initialize(project='timdata')

df = pd.read_csv("data/exported/nora.csv")
df.columns = df.columns.str.strip()

min_lon = df["lon"].min()
max_lon = df["lon"].max()
min_lat = df["lat"].min()
max_lat = df["lat"].max()

n_rows = 4
n_cols = 4

lon_steps = [(min_lon + i * (max_lon - min_lon) / n_cols) for i in range(n_cols + 1)]
lat_steps = [(min_lat + i * (max_lat - min_lat) / n_rows) for i in range(n_rows + 1)]

collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat]))
    .filterDate("2022-03-26", "2022-04-07")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
)

image = collection.first()
bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

output_rows = []
batch_id = 0

for i in range(n_rows):
    for j in range(n_cols):
        lon_min = lon_steps[j]
        lon_max = lon_steps[j + 1]
        lat_min = lat_steps[i]
        lat_max = lat_steps[i + 1]

        tile_df = df[
            (df["lon"] >= lon_min) & (df["lon"] < lon_max) &
            (df["lat"] >= lat_min) & (df["lat"] < lat_max)
        ]

        if tile_df.empty:
            continue

        features = []
        for idx, row in tile_df.iterrows():
            point = ee.Geometry.Point(float(row["lon"]), float(row["lat"]))
            feat = ee.Feature(point).set({
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                "som": float(row["som"])
            })
            features.append(feat)

        fc = ee.FeatureCollection(features)

        sampled = image.select(bands).sampleRegions(
            collection=fc,
            scale=10,
            geometries=False
        )

        sampled_list = sampled.getInfo()["features"]

        for feature in sampled_list:
            props = feature["properties"]
            row = {
                "lon": props["lon"],
                "lat": props["lat"],
                "som": props["som"]
            }
            for band in bands:
                row[band] = props.get(band, np.nan)
            output_rows.append(row)

        tile_geom = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
        task = ee.batch.Export.image.toCloudStorage(
            image=image.select(bands),
            description=f"export_tile_{batch_id}",
            bucket="timdata-bucket",
            fileNamePrefix=f"data/exported/nora_tile_{batch_id}",
            region=tile_geom,
            scale=10,
            fileFormat="GeoTIFF",
            maxPixels=1e9
        )
        task.start()
        print(f"Submitted tile {batch_id}, {len(tile_df)} points")
        batch_id += 1

os.makedirs("data/exported", exist_ok=True)
out_df = pd.DataFrame(output_rows)
out_df.to_csv("data/exported/nora.csv", index=False)
print("Combined CSV written to data/exported/nora.csv")
