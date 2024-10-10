import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random


def self_relu(x):
    eps = -0.01
    return torch.where(x > 0, x, eps)


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 1)
        self.activation = self_relu

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x


def train(model, loader, criterion, optimizer, epochs=25):
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
        rdn = random.randint(0, 2000)
        rdn_bias = random.randint(1,5)
        data.append([rdn, (rdn + 1) * 2 + rdn_bias])
    return data


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.scale = torch.max(self.data)
        self.data = self.data / self.scale

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
            # print(f'x: {x}| y_est: {output} | y_true: {y}')
            total += y.size(0)
            correct += (abs(output - y) < 0.02).sum().item()

    acc = 100 * correct / total

    if acc > 99:
        torch.save(model.state_dict(), 'model.pth')

    print(f'Accuracy: {acc:.2f}%')

    return acc


def main():
    model = SimpleNN()
    dataset = generate_dataset()
    train_model = False
    print_loader = False

    accs = []
    test_nb = 1000

    loader = DataLoader(CustomDataset(dataset), batch_size=64, shuffle=True)

    if train_model:
        criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train(model, loader, criterion, optimizer)
    else:
        model.load_state_dict(torch.load('model.pth', weights_only=True))

    for _ in range(test_nb):
        loader = DataLoader(CustomDataset(dataset), batch_size=64, shuffle=True)
        if print_loader:
            for x, y in loader:
                print(f'Input x: ', end="")
                for el in x:
                    print(f"{el.item()} ", end="")
                print("")

                print(f'Input y: ', end="")
                for el in y:
                    print(f"{el.item()} ", end="")
                print("")

        accs.append(test(model, loader))
    accs = torch.Tensor(accs)
    print(f'Mean Accuracy: {accs.mean():.2f}%')
    print(f'Standard Deviation: {accs.std():.2f}%')


main()
