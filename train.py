from datasets import load_dataset
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
from models.resnet import ResNet

# Load the dataset
ds = load_dataset("uoft-cs/cifar10")


# Define the transform function


def transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((72, 72)),
        transforms.RandomCrop(64, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

# Define the SimpleDataLoader class


class SimpleDataLoader:
    def __init__(self, ds, transform):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.transform(self.ds[idx]['img']), self.ds[idx]['label']


# Create train and test datasets
train_ds = SimpleDataLoader(ds["train"], transform())
test_ds = SimpleDataLoader(ds["test"], transform())

# Create data loaders
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)


def train(epochs, logging_step, lr, batch_size, model, device):
    # Initialize the model, loss function, and optimizer
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % logging_step == 0:
                print(f"Epoch :{epoch+1}, Step: {i+1}, Loss: {loss.item()}")

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        # Evaluation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(
                f"Accuracy of the network on the 10000 test images: {100 * correct / total}%")


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.append(".")
    from models.simple_cnn import SimpleCNN

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--logging_step", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--model", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    epochs = args.epochs
    logging_step = args.logging_step
    lr = args.lr
    batch_size = args.batch_size
    device = args.device
    model = None
    if args.model == "simple_cnn":
        model = SimpleCNN()
    elif args.model == "resnet":
        model = ResNet()
    else:
        raise ValueError(f"Model {args.model} not found")
    # model parameters
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {model_params}")
    train(epochs, logging_step, lr, batch_size, model, device)
