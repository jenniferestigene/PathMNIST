"""
evaluate.py

Evaluates the trained TissueTypeCNN on the held-out PathMNIST test set
(7,180 images, never seen during training). 

Reports overall accuracy, per-class precision/recall/F1,
and saves a 9x9 confusion matrix to assets/confusion_matrix.png.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from model import TissueTypeCNN
from dataset import CLASS_NAMES
from train import PathMNISTDataset


image_size = 64

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test set 
    test_dataset = PathMNISTDataset(f"tissue_test_images_{image_size}.npy", f"tissue_test_labels_{image_size}.npy", train=False)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # Load the trained weights into a fresh model instance
    model = TissueTypeCNN(num_classes=9)
    model.load_state_dict(torch.load("saved_model.pth", map_location=device))
    model.to(device)

    # eval() mode disables Dropout and switches BatchNorm to use its
    # running statistics instead of the current batch's
    # required for consistent, deterministic predictions at inference time.
    model.eval()

    all_predictions = []
    all_labels = []

    # No gradient tracking needed
    # torch.no_grad() saves memory and compute.
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            # .cpu() before .numpy(): sklearn's metrics functions expect
            # NumPy arrays on CPU, not GPU tensors — this line is a no-op
            # if we're already on CPU, but required if device is "cuda".
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Overall accuracy
    test_accuracy = (all_predictions == all_labels).mean()
    print(f"\nTest accuracy: {test_accuracy:.4f}\n")

    # Per-class precision, recall, F1
    report = classification_report(all_labels, all_predictions, target_names=CLASS_NAMES, digits=4)
    print("Classification report:")
    print(report)

    # Save the report to a text file so it's easy to paste into results.md
    # without re-running evaluation.
    with open("classification_report.txt", "w") as f:
        f.write(f"Test accuracy: {test_accuracy:.4f}\n\n")
        f.write(report)
    print("Saved classification_report.txt")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("PathMNIST — Confusion Matrix (Test Set)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()

    fig.savefig("assets/confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("Saved assets/confusion_matrix.png")


if __name__ == "__main__":
    evaluate()