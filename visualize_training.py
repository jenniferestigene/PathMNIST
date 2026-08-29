"""
visualize_training.py

Builds assets/training_curves.png from training_log.csv, a two-panel
figure (loss, accuracy) with train and val curves overlaid on each.
"""

import pandas as pd
import matplotlib.pyplot as plt


def plot_training_curves(log_path: str, out_path: str):
    df = pd.read_csv(log_path)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Loss panel ---
    ax_loss.plot(df["epoch"], df["train_loss"], label="Train Loss")
    ax_loss.plot(df["epoch"], df["val_loss"], label="Val Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training and Validation Loss")
    ax_loss.legend()

    # --- Accuracy panel ---
    ax_acc.plot(df["epoch"], df["train_acc"], label="Train Accuracy")
    ax_acc.plot(df["epoch"], df["val_acc"], label="Val Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Training and Validation Accuracy")
    ax_acc.legend()

    fig.suptitle("PathMNIST — TissueTypeCNN Training Curves")
    fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved training curves to {out_path}")

    # Last epoch's stats — useful for spotting overfitting/instability at
    # the end of training, but NOT what's actually saved to saved_model.pth.
    print(f"Final epoch ({int(df['epoch'].iloc[-1])}) — "
          f"train_acc: {df['train_acc'].iloc[-1]:.4f}, "
          f"val_acc: {df['val_acc'].iloc[-1]:.4f}, "
          f"val_loss: {df['val_loss'].iloc[-1]:.4f}")

    # Best-val-loss epoch's stats — this is the checkpoint train.py actually
    # writes to saved_model.pth, so this is the number that matches what
    # evaluate.py will load. PathMNIST's LR schedule (ReduceLROnPlateau)
    # means the best epoch is often not the last one — worth reporting
    # both rather than assuming they're the same.
    best_row = df.loc[df["val_loss"].idxmin()]
    print(f"Best epoch ({int(best_row['epoch'])}, lowest val_loss, "
          f"matches saved_model.pth) — "
          f"train_acc: {best_row['train_acc']:.4f}, "
          f"val_acc: {best_row['val_acc']:.4f}, "
          f"val_loss: {best_row['val_loss']:.4f}")


if __name__ == "__main__":
    plot_training_curves("training_log.csv", "assets/training_curves.png")