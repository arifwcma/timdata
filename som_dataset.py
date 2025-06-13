import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class SomDataset(Dataset):
    def __init__(self, is_train=True):
        df = pd.read_csv("data/exported/vectis_full.csv")
        df = df.dropna()
        bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
        X = df[bands].values.astype("float32")
        y = df["som"].values.astype("float32")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        self.X = torch.tensor(X_train if is_train else X_test)
        self.y = torch.tensor(y_train if is_train else y_test).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
