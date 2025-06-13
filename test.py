import torch
from torch.utils.data import DataLoader
from som_dataset import SomDataset
from som_model import SomModel
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def test():
    model = SomModel()
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    test_dataset = SomDataset(is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    criterion = nn.MSELoss()
    total_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X, y in test_loader:
            output = model(X)
            loss = criterion(output, y)
            total_loss += loss.item()
            y_true.extend(y.numpy())
            y_pred.extend(output.numpy())

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"Test MSE: {total_loss:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")


if __name__ == "__main__":
    test()
