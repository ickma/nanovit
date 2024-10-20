
import torch
from torch import nn
from torch.nn import functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # output is (6,32,32)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)  # input is (BatchSize, 3, 64, 64)
        self.max_pool1 = nn.MaxPool2d(2, 2)  # output is (32,32,32)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)  # output is (64,28,28)
        self.max_pool2 = nn.MaxPool2d(2, 2)  # output is (64,14,14)
        self.bn2 = nn.BatchNorm2d(64)
        self.avg_pool = nn.AvgPool2d(2, 2)  # output is (64,7,7)
        self.fc1 = nn.Linear(64 * 7 * 7, 120)
        self.dropout = nn.Dropout(0.2)

        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.max_pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.max_pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
