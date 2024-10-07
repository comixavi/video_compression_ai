import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Conv2DNN(nn.Module):
    def __init__(self):
        super(Conv2DNN, self).__init__()
        self.fc1 = nn.Conv2d(1, 4, 4)
        self.fc2 = nn.Flatten()
        # For an input of size (H_in, W_in) with:
        # - Kernel size (K)
        # - Padding (P)
        # - Stride (S)
        # - Dilation (D)
        # The output size (H_out, W_out) is computed as:
        # H_out = floor( (H_in + 2 * P - D * (K - 1) - 1) / S ) + 1
        # W_out = floor( (W_in + 2 * P - D * (K - 1) - 1) / S ) + 1
        self.fc3 = nn.Linear(4 * 25 * 25, 128)
        self.fc4 = nn.Linear(128, 10)
        self.activation = torch.relu

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x


class Conv1DNN(nn.Module):
    def __init__(self):
        super(Conv1DNN, self).__init__()

        self.layers = nn.ModuleList([
            nn.Conv1d(1, 4, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3124, 10)
        ])

    def forward(self, x):
        # the input for conv1d needs converting to 3D (ironically);
        x = x.view(x.size(0), 1, -1)
        for layer in self.layers:
            x = layer(x)

        return x


def self_flatten_no_rec(x: torch.Tensor) -> torch.Tensor:
    flattened = []

    stack = [x]

    while stack:
        current = stack.pop()

        if current.dim() == 0:
            flattened.append(current.item())
        else:
            stack.extend(reversed(current))
    return torch.tensor(flattened, dtype=x.dtype).reshape(x.size(0), -1)


def self_flatten_rec(x: torch.Tensor):
    def base(dim_n, out=None):
        if out is None:
            out = []

        if dim_n.dim() == 0:
            out.append(dim_n.item())
            return out

        for dim_n_1 in dim_n:
            base(dim_n_1, out)

        return out

    return torch.tensor(base(x)).reshape(x.size(0), -1)


class LinearNN(nn.Module):

    def self_relu(self, x):
        return torch.tensor([(el if el > 0 else 0) for el in x])

    def __init__(self):
        super(LinearNN, self).__init__()
        self.flatten = self_flatten_no_rec
        self.fc1 = nn.Linear(28 * 28, 32)
        self.fc2 = nn.Linear(32, 10)
        self.activation = torch.relu

    def forward(self, x):
        try:
            x = self_flatten_no_rec(x)
            x = self.activation(self.fc1(x))
            x = self.fc2(x)
        except:
            print(x.shape)
        return x


def train(model, loader, criterion, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(loader):.4f}')


def test(model, loader, show_figs=False):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            if show_figs:
                print(torch.softmax(outputs, dim=1))
                tfig = torchvision.utils.make_grid(images)
                tfig = tfig / 2 + 0.5
                npimg = tfig.numpy()
                plt.imshow(np.transpose(npimg, (1, 2, 0)))
                plt.show()
                input()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (0.25,))])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

model = LinearNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(model.parameters(), lr=0.01)
train(model, train_loader, criterion, optimizer)
test(model, test_loader)
