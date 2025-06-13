import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from som_dataset import SomDataset
from som_model import SomModel

def train():
    model = SomModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = SomDataset(is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(50):
        total_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/50 - Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "model.pt")
