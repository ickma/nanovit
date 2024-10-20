import torch
from torch import nn
from torch.nn import functional as F


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size, padding=padding, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # skip connection
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, bias=False, stride=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        output = F.relu(self.bn1(self.conv1(x)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.relu(output + self.skip(x))
        output = F.max_pool2d(output, 2, 2)
        output = F.dropout(output, 0.2)
        return output


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # input is (BatchSize, 3, 64, 64), output is (BatchSize, 64, 32, 32)
        self.block1 = ResNetBlock(3, 64)
        # input is (BatchSize, 64, 32, 32), output is (BatchSize, 128, 16, 16)
        self.block2 = ResNetBlock(64, 128)
        # input is (BatchSize, 128, 16, 16), output is (BatchSize, 256, 8, 8)
        self.block3 = ResNetBlock(128, 256)
        # input is (BatchSize, 256, 8, 8), output is (BatchSize, 512, 4, 4)
        self.block4 = ResNetBlock(256, 512)

    def forward(self, x):
        output = self.block1(x)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)

        return output


class ResNetModel(nn.Module):
    def __init__(self, channels=512, length=16, num_classes=10):
        super().__init__()
        self.resnet = ResNet()
        self.fc = nn.Linear(channels * length, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        output = self.resnet(x)
        output = torch.flatten(output, 1)
        output = self.dropout(output)
        output = self.fc(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output
