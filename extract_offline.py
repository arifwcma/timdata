import pandas as pd
import rasterio
import numpy as np
from pyproj import Transformer
import os

df = pd.read_csv("data/exported/vectis.csv", encoding="ISO-8859-1")
df.columns = df.columns.str.strip()

tif_path = "data/exported/sentinel_image.tif"
bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

with rasterio.open(tif_path) as src:
    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    xy_proj = [transformer.transform(lon, lat) for lon, lat in zip(df["lon"], df["lat"])]
    values = list(src.sample(xy_proj))

values = np.array(values)

for i, band in enumerate(bands):
    df[band] = values[:, i]

os.makedirs("data/exported", exist_ok=True)
df.to_csv("data/exported/vectis_full_offline.csv", index=False)

print("vectis_full_offline.csv written")
