import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class SomDataset(Dataset):
    def __init__(self, is_train=True, aux=None):
        df = pd.read_csv("data/exported/vectis_full.csv").dropna()
        bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
        features = bands + (aux if aux else [])

        X = df[features].values.astype("float32")
        y = df["som"].values.astype("float32")

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.25, random_state=42
        )

        self.X = torch.tensor(X_train if is_train else X_test)
        self.y = torch.tensor(y_train if is_train else y_test).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
