"""
visualize_misclassifications.py

Pulls specific misclassified test examples for visual inspection, to
distinguish between failure modes (e.g. genuinely confusable tissue vs.
a systematic domain-shift artifact like staining variation) that a
confusion matrix alone can't diagnose.

Default: the adipose -> smooth muscle confusion pair, since it accounts
for ~35% of all test-set errors (see classification_report.txt / 
confusion_matrix.png). Shows misclassified adipose examples alongside
correctly-classified adipose and smooth muscle examples for comparison.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import TissueTypeCNN
from dataset import CLASS_NAMES
from train import PathMNISTDataset

image_size = 64


def _to_displayable(image_tensor):
    """
    Convert a (3, H, W) float32 tensor in [0, 1] to a (H, W, 3) numpy
    array matplotlib's imshow can render.
    """
    return image_tensor.permute(1, 2, 0).cpu().numpy()


def _plot_grid(images, title, out_path, n_cols=8):
    n = len(images)
    n_cols = min(n_cols, max(n, 1))
    n_rows = max(1, -(-n // n_cols))  # ceil division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 1.8))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols, i % n_cols]
        ax.axis("off")
        if i < n:
            ax.imshow(images[i])

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def visualize_confusion(true_class: str, pred_class: str, n: int = 8):
    """
    Loads the test set and trained model, then saves three comparison
    grids to assets/:
      1. Misclassified examples: true_class predicted as pred_class
      2. Correctly classified true_class examples, for reference
      3. Correctly classified pred_class examples, for reference

    Args:
        true_class: ground-truth class name (must be in CLASS_NAMES).
        pred_class: predicted class name the model confused it with.
        n: max number of examples to show per grid.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    true_idx = CLASS_NAMES.index(true_class)
    pred_idx = CLASS_NAMES.index(pred_class)

    test_dataset = PathMNISTDataset(
        f"tissue_test_images_{image_size}.npy", f"tissue_test_labels_{image_size}.npy", train=False
    )
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    model = TissueTypeCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load("saved_model.pth", map_location=device))
    model.to(device)
    model.eval()

    misclassified = []  # true_class images predicted as pred_class
    correct_true = []   # true_class images correctly predicted
    correct_pred = []   # pred_class images correctly predicted

    with torch.no_grad():
        for images, labels in test_loader:
            images_dev = images.to(device)
            outputs = model(images_dev)
            predictions = outputs.argmax(dim=1).cpu()

            for img, label, pred in zip(images, labels, predictions):
                label, pred = label.item(), pred.item()

                if label == true_idx and pred == pred_idx and len(misclassified) < n:
                    misclassified.append(_to_displayable(img))
                elif label == true_idx and pred == true_idx and len(correct_true) < n:
                    correct_true.append(_to_displayable(img))
                elif label == pred_idx and pred == pred_idx and len(correct_pred) < n:
                    correct_pred.append(_to_displayable(img))

            if len(misclassified) >= n and len(correct_true) >= n and len(correct_pred) >= n:
                break

    print(f"Found {len(misclassified)} examples of {true_class} misclassified as {pred_class}")

    _plot_grid(
        misclassified,
        f"True: {true_class} | Predicted: {pred_class} (misclassified)",
        f"assets/misclassified_{true_class.replace(' ', '_')}_as_{pred_class.replace(' ', '_')}.png",
    )
    _plot_grid(
        correct_true,
        f"Correctly classified: {true_class}",
        f"assets/correct_{true_class.replace(' ', '_')}.png",
    )
    _plot_grid(
        correct_pred,
        f"Correctly classified: {pred_class}",
        f"assets/correct_{pred_class.replace(' ', '_')}.png",
    )


if __name__ == "__main__":
    # The dominant confusion pair from the confusion matrix: 502 of 1,338
    # adipose test images (37.5%) were predicted as smooth muscle.
    visualize_confusion(true_class="adipose", pred_class="smooth muscle", n=8)

    # New dominant confusion pair, per the updated confusion matrix:
    # 111 of 421 cancer-associated stroma test images (26.4%) were
    # predicted as debris, and debris more broadly absorbs stray
    # misclassifications from several other classes (smooth muscle,
    # mucus, normal colon mucosa, colorectal adenocarcinoma epithelium),
    # suggesting debris may be acting as a "catch-all" class rather than
    # a clean visual confusion with one specific pair.
    visualize_confusion(true_class="cancer-associated stroma", pred_class="debris", n=8)