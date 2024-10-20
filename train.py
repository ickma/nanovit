from datasets import load_dataset
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd

from models.simple_cnn import SimpleCNNModel, SimpleCNN
from models.resnet import ResNetModel, ResNet
from models.vit import ViT
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


def train(epochs, logging_step, lr, batch_size, model, device,model_name):
    # Initialize the model, loss function, and optimizer
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)


    # Training loop
    for epoch in range(epochs):
        loss_list = []
        acc_list = []
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append([epoch+1, i+1, loss.item()])
            loss.backward()
            optimizer.step()
            if i % logging_step == 0:
                print(f"Epoch :{epoch+1}, Step: {i+1}, Loss: {loss.item()}")

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        write_mode = 'w' if epoch == 0 else 'a'
        pd.DataFrame(loss_list, columns=["Epoch", "Step", "Loss"]).to_csv(
            f"{model_name}_loss.csv", index=False, header=write_mode == 'w', mode=write_mode)

        # Evaluation
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc_list.append([epoch+1, correct/total])
            print(
                f"Accuracy of the network on the 10000 test images: {100 * correct / total}%")
            pd.DataFrame(acc_list, columns=["Epoch", "Accuracy"]).to_csv(
                f"{model_name}_acc.csv", index=False, header=write_mode == 'w', mode=write_mode)


if __name__ == "__main__":
    import argparse

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
        model = SimpleCNNModel()
    elif args.model == "resnet":
        model = ResNetModel(channels=512, length=16)
    elif args.model == "simple_vit":
        model = ViT(cnn_model_cls=SimpleCNN,
                    input_channels=64, input_len=64, heads=8)
    elif args.model == "res_vit":
        model = ViT(cnn_model_cls=ResNet,
                    input_channels=512, input_len=16, heads=2)
    elif args.model == "resnet_model":
        raise ValueError(f"Model {args.model} not found")
    # model parameters
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {model_params}")
    train(epochs, logging_step, lr, batch_size, model, device, args.model)
