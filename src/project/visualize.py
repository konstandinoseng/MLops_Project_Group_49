"""Training with train/test split and accuracy plotting."""

from pathlib import Path
from typing import List, Literal, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]

from project.data import FinancialPhraseBankDataset
from project.model import TextSentimentModel


def train_with_plot(
    root_path: str,
    agreement: Literal["AllAgree", "75Agree", "66Agree", "50Agree"] = "AllAgree",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    save_path: Optional[str] = None,
    plot_path: Optional[str] = None,
) -> Tuple[List[float], List[float]]:
    """Train with train/val split and return (train_accs, val_accs)."""
    ds = FinancialPhraseBankDataset(root_path, agreement=agreement)
    vocab = ds.build_vocab(min_freq=1)

    # Split into train/val
    n = len(ds)
    indices = torch.randperm(n).tolist()
    split = int(n * (1 - val_split))
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=ds.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=ds.collate_fn)

    model = TextSentimentModel(vocab_size=len(vocab), embedding_dim=64, num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_accs: List[float] = []
    val_accs: List[float] = []

    for epoch in range(epochs):
        # Train
        model.train()
        correct, total = 0, 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)
            correct += int((preds == targets).sum().item())
            total += int(targets.size(0))
        train_acc = correct / max(total, 1)
        train_accs.append(train_acc)

        # Validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                logits = model(inputs)
                preds = logits.argmax(dim=1)
                correct += int((preds == targets).sum().item())
                total += int(targets.size(0))
        val_acc = correct / max(total, 1)
        val_accs.append(val_acc)

        print(f"epoch={epoch+1:02d} | train_acc={train_acc:.3f} | val_acc={val_acc:.3f}")

    # Save model
    out_path = Path(save_path) if save_path else Path("models") / f"text_model_{agreement}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab_size": len(vocab)}, out_path)
    print(f"Saved model to {out_path}")

    # Plot
    if plt is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), train_accs, label="Train Accuracy", marker="o", markersize=3)
        plt.plot(range(1, epochs + 1), val_accs, label="Validation Accuracy", marker="s", markersize=3)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Train vs Validation Accuracy ({agreement})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_path = Path(plot_path) if plot_path else Path("reports/figures") / f"accuracy_plot_{agreement}.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=150)
        print(f"Saved plot to {fig_path}")
        plt.close()
    else:
        print("matplotlib not installed. Skipping plot.")

    return train_accs, val_accs


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else r"F:\Business Analytics Dk\MLOps\FinancialPhraseBank-v1.0"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    train_with_plot(path, agreement="AllAgree", epochs=epochs)
