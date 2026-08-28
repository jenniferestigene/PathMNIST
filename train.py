"""
train.py

Trains TissueTypeCNN on the PathMNIST dataset.

Uses inverse-frequency class weighting to correct for the 1.63x imbalance
between the largest class (colorectal adenocarcinoma epithelium) and 
smallest (normal colon mucosa) documented in assets/class_distribution.png.

Device handling automatically uses GPU if available (torch.cuda.is_available()),
falling back to CPU otherwise.
"""

import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import TissueTypeCNN

image_size = 64

class PathMNISTDataset(Dataset):
    """
    Wraps cached PathMNIST .npy tensors for use with
    PyTorch's DataLoader.

    Augmentation is applied only when train=True.

    Reasoning: histology tissue patches have no canonical orientation, 
    so flipping/rotating a patch doesn't change its true label.
    """

    def __init__(self, images_path: str, labels_path: str, train: bool = False):
        self.images = torch.from_numpy(np.load(images_path))
        self.labels = torch.from_numpy(np.load(labels_path))

        if train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label



def compute_class_weights(labels: np.ndarray, num_classes: int = 9) -> torch.Tensor:
    """
    Compute inverse-frequency clqss weights for CrossEntropyLoss, correcting 
    for PathMNIST's 1.63x imbalance between the largest class (colorectal
    adenocarcinoma epithelium) and smallest (normal colon mucosa) 
    - see assets/class_distribution.png

    weight[c] = total_samples / (num_classes * count[c])

    This will keep the average weight near 1.0, so it rescales the relative
    importance of classes without inflating the overall loss magnitude.

    Args: 
        labels: np.ndarray of integer class indices from the training set.
        num_classes: total number of classes

    Returns: 
        torch.Tensor of shape (num_classes), dtype float32, ready to pass
        as CrossEntropyLoss(weight=...).
    """

    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)

    weights = total / (num_classes * counts)

    return torch.tensor(weights, dtype=torch.float32)



def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using ddevice: {device}")

    # Datasets and loaders
    train_dataset = PathMNISTDataset(
        f"tissue_train_images_{image_size}.npy", f"tissue_train_labels_{image_size}.npy", train=True
    )
    val_dataset = PathMNISTDataset(
        f"tissue_val_images_{image_size}.npy", f"tissue_val_labels_{image_size}.npy", train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

    # Model
    model = TissueTypeCNN(num_classes=9)
    model.to(device)

    # Weighted loss
    train_labels = np.load(f"tissue_train_labels_{image_size}.npy")
    class_weights = compute_class_weights(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # LR Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

    # Logging
    log_rows = []
    num_epochs = 20

    for epoch in range(1, num_epochs +1):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation 
        model.eval()
        val_loss, val_correct, val_total = 0.00, 0, 0


        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                predictions = outputs.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Step the scheduler
        scheduler.step(val_loss)

        # Read current LR for logging
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:2d}/{num_epochs} | "
              f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f} val_acc: {val_acc}:.4f | "
              f"lr: {current_lr:.6f}")

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        })

    # Write training log to CSV
    with open("training_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f,
                                fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"],
                                )
        writer.writeheader()
        writer.writerows(log_rows)

    print("Saved training_log.csv")

    torch.save(model.state_dict(), "saved_model.pth")
    print("Saved saved_model.pth")


if __name__ == "__main__":
    train()