import ee
import pandas as pd
import os

ee.Initialize(project='timdata')

df = pd.read_csv("https://storage.googleapis.com/timdata-bucket/vectis.csv", encoding="ISO-8859-1")
df.columns = df.columns.str.strip()

min_lon = df["lon"].min()
max_lon = df["lon"].max()
min_lat = df["lat"].min()
max_lat = df["lat"].max()
bbox = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(bbox)
    .filterDate("2022-03-26", "2022-04-07")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
)

image = collection.first()

bands = [
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"
]

features = []
for i, row in df.iterrows():
    point = ee.Geometry.Point(float(row["lon"]), float(row["lat"]))
    feat = ee.Feature(point).set("index", i)
    features.append(feat)

fc = ee.FeatureCollection(features)

sampled = image.select(bands).sampleRegions(
    collection=fc,
    scale=10,
    geometries=False
)

sampled_dicts = sampled.aggregate_array("index").getInfo()
band_data = [sampled.aggregate_array(band).getInfo() for band in bands]

for i, band in enumerate(bands):
    df[band] = [row[i] if i < len(row) else None for row in zip(*band_data)]

os.makedirs("data/exported", exist_ok=True)
df.to_csv("data/exported/vectis_full.csv", index=False)

task = ee.batch.Export.image.toCloudStorage(
    image=image.select(bands),
    description="export_sentinel_image",
    bucket="timdata-bucket",
    fileNamePrefix="data/exported/sentinel_image",
    region=bbox,
    scale=10,
    fileFormat="GeoTIFF",
    maxPixels=1e9
)
task.start()

print("vectis_full.csv written")
print("sentinel_image.tif export task submitted")
