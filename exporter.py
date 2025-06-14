import pandas as pd

df = pd.read_csv("data/original/vectis.csv", encoding="ISO-8859-1")

df_selected = df[[
    'Longitude',
    'Latitude',
    'Time',
    'Elevation(m)',
    'Soil OM(%)',
    'Soil Mois(%)',
    'Soil Temp(°C)'
]].rename(columns={
    'Longitude': 'lon',
    'Latitude': 'lat',
    'Time': 'time',
    'Elevation(m)': 'elevation',
    'Soil OM(%)': 'som',
    'Soil Mois(%)': 'moisture',
    'Soil Temp(°C)' : 'temp'
})

df_selected.to_csv("data/exported/vectis.csv", index=False)
