import rasterio
import pandas as pd
import numpy as np
import requests
import os

url = "https://storage.googleapis.com/timdata-bucket/data/exported/sentinel_image.tif"
tif_path = "data/exported/sentinel_image.tif"
csv_path = "data/exported/sentinel_pixels.csv"

os.makedirs("data/exported", exist_ok=True)

if not os.path.exists(tif_path):
    r = requests.get(url)
    with open(tif_path, "wb") as f:
        f.write(r.content)

with rasterio.open(tif_path) as src:
    bands = [src.read(i + 1) for i in range(src.count)]
    transform = src.transform
    width, height = src.width, src.height
    band_names = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

    data = []
    for row in range(height):
        for col in range(width):
            values = [band[row, col] for band in bands]
            if all(np.isnan(v) or v == 0 for v in values):
                continue
            x, y = rasterio.transform.xy(transform, row, col)
            data.append([y, x] + values)

df = pd.DataFrame(data, columns=["lat", "lon"] + band_names)
df.to_csv(csv_path, index=False)

print("sentinel_pixels.csv written")
