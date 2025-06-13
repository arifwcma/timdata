import pandas as pd
import matplotlib.pyplot as plt

def plot_spectral_curve(n):
    df = pd.read_csv("data/exported/vectis_full.csv")
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    values = df.loc[n, bands].values.astype(float)

    plt.figure(figsize=(8, 4))
    plt.plot(bands, values, marker='o')
    plt.title(f"Spectral Curve for Row {n}")
    plt.xlabel("Band")
    plt.ylabel("Reflectance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

for i in range(0,100,20):
    plot_spectral_curve(i)