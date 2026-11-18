import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

INPUT_SIZE = 28 * 28  # 784 pixels per image
HIDDEN_SIZE = 32
OUTPUT_SIZE = 10


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE, bias=False),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE, bias=False),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)


def get_dataloaders(batch_size=64, root="data"):
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    training_data = datasets.MNIST(
        root=root, train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root=root, train=False, download=True, transform=transform
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader


def get_device():
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator().type
    return "cpu"
