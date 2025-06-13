import torch
from torch.utils.data import DataLoader
from som_dataset import SomDataset
from som_model import SomModel
import torch.nn as nn

def test():
    model = SomModel()
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    test_dataset = SomDataset(is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    criterion = nn.MSELoss()
    total_loss = 0

    with torch.no_grad():
        for X, y in test_loader:
            output = model(X)
            loss = criterion(output, y)
            total_loss += loss.item()

    print(f"Test MSE: {total_loss:.4f}")
