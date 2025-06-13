import pandas as pd

df = pd.read_csv("vectis.csv", encoding="ISO-8859-1")

df_selected = df[[
    'Longitude',
    'Latitude',
    'Time',
    'Elevation(m)',
    'Soil OM(%)',
    'Soil Mois(%)'
]].rename(columns={
    'Longitude': 'lon',
    'Latitude': 'lat',
    'Time': 'time',
    'Elevation(m)': 'elevation',
    'Soil OM(%)': 'som',
    'Soil Mois(%)': 'moisture'
})

df_selected.to_csv("vectis_filtered.csv", index=False)
