from datetime import datetime
import time
from datasets import load_dataset
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd

from models.simple_cnn import SimpleCNNModel
from models.resnet import ResNetModel
from models.vit import ViT
# Load the dataset
ds = load_dataset("uoft-cs/cifar10")


# Define the transform function


def transform(eval=False, img_size=128):
    if eval:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((int(img_size*1.1), int(img_size*1.1))),
        transforms.RandomCrop(img_size, padding=4),
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
test_ds = SimpleDataLoader(ds["test"], transform(eval=True))

# Create data loaders
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)


def train(model, training_args):
    print(training_args)
    model.to(training_args["device"])
    # Initialize the model, loss function, and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=training_args["lr"],
                          momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=training_args["epochs"]//5, gamma=0.5, verbose=True)
    if training_args["model"].find("vit") != -1:
        optimizer = optim.AdamW(
            model.parameters(), lr=training_args["lr"], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_args["epochs"], eta_min=training_args["lr"]/100)

    # Create train and test datasets
    train_ds = SimpleDataLoader(
        ds["train"], transform(img_size=training_args["img_size"]))
    test_ds = SimpleDataLoader(
        ds["test"], transform(eval=True, img_size=training_args["img_size"]))

    # Create data loaders
    train_loader = DataLoader(
        train_ds, batch_size=training_args["batch_size"], shuffle=True)
    test_loader = DataLoader(
        test_ds, batch_size=training_args["batch_size"], shuffle=False)

    # Training loop
    for epoch in range(training_args["epochs"]):
        start_time = time.time()
        loss_list = []
        acc_list = []
        latency_list = []
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(
                training_args["device"]), labels.to(training_args["device"])
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append([epoch+1, i+1, loss.item()])
            loss.backward()
            optimizer.step()
            if i % training_args["logging_step"] == 0:
                print(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Epoch :{epoch+1}, Step: {i+1}, Loss: {loss.item()}")

        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Epoch {epoch+1}, Loss: {loss.item()}")
        write_mode = 'w' if epoch == 0 else 'a'
        pd.DataFrame(loss_list, columns=["Epoch", "Step", "Loss"]).to_csv(
            f"{training_args['model']}_loss.csv", index=False, header=write_mode == 'w', mode=write_mode)
        scheduler.step()

        # Evaluation
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for images, labels in test_loader:
                images, labels = images.to(
                    training_args["device"]), labels.to(training_args["device"])
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc_list.append([epoch+1, correct/total])
            end_time = time.time()
            latency_list.append([epoch+1, end_time-start_time])
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Accuracy of the network on the {len(test_ds)} test images: {100 * correct / total}%")
            pd.DataFrame(acc_list, columns=["Epoch", "Accuracy"]).to_csv(
                f"{training_args['model']}_acc.csv", index=False, header=write_mode == 'w', mode=write_mode)
            pd.DataFrame(latency_list, columns=["Epoch", "Latency"]).to_csv(
                f"{training_args['model']}_latency.csv", index=False, header=write_mode == 'w', mode=write_mode)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--logging_step", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--model", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--embed_size", type=int, default=128)
    args = parser.parse_args()

    epochs = args.epochs
    logging_step = args.logging_step
    lr = args.lr
    batch_size = args.batch_size
    device = args.device
    img_size = args.img_size
    embed_size = args.embed_size
    model = None
    if args.model == "simple_cnn":
        model = SimpleCNNModel()
    elif args.model == "resnet":
        model = ResNetModel(channels=512, length=16)
    elif args.model == "vit":  # vit, linear projection , scaled dot product attention
        # we need 16*16 patches,adjust as needed
        model = ViT(patch_size=img_size//16, linear_proj=True,
                    img_size=img_size, emd_size=embed_size)
    elif args.model == "vitl":  # vil, linear projection, linear attention
        model = ViT(linear_attn=True, patch_size=1,
                    img_size=img_size, emd_size=embed_size)
    elif args.model == "vitx":  # vix, linear projection, cnn projection
        model = ViT(cnn_proj=True,
                    linear_attn=True,
                    linear_proj=False,
                    patch_size=1,
                    img_size=img_size, emd_size=embed_size)
    else:
        raise ValueError(f"Model {args.model} not found")
    # model parameters
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {model_params}")
    train(model, args.__dict__)
