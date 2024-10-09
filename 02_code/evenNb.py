import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random


def self_relu(x):
    eps = 0.001
    return torch.where(x > 0, x, eps)


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(8, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x


def train(model, loader, criterion, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0

        for x, y in loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(loader):.4f}')


def generate_dataset():
    data = []
    nbGen = 1024
    for _ in range(nbGen):
        rdn = random.randint(0, 1000)
        data.append([rdn * 2, (rdn + 1) * 2])
    return data


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor([x], dtype=torch.float32), torch.tensor([y], dtype=torch.float32)


def test(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for (x, y) in loader:
            output = model(x)
            output = output.round()  # Round the output for comparison
            print(f'x: {x}| y_est: {output} | y_true: {y}')
            total += y.size(0)
            correct += (output == y).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')


def main():
    model = SimpleNN()
    dataset = generate_dataset()
    loader = DataLoader(CustomDataset(dataset), batch_size=8, shuffle=True)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train(model, loader, criterion, optimizer)
    test(model, loader)


main()
