import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from project.data import FinancialPhraseBankDataset
from project.model import TextSentimentModel


def train_phrasebank(
    root_path: str,
    agreement: str = "AllAgree",
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> None:
    ds = FinancialPhraseBankDataset(root_path, agreement=agreement)  # e.g., F:\Business Analytics Dk\MLOps\FinancialPhraseBank-v1.0
    vocab = ds.build_vocab(min_freq=1)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=ds.collate_fn)

    model = TextSentimentModel(vocab_size=len(vocab), embedding_dim=64, num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            preds = logits.argmax(dim=1)
            correct += int((preds == targets).sum().item())
            total += int(targets.size(0))
        acc = correct / max(total, 1)
        print(f"phrasebank({agreement}) | epoch={epoch+1} loss={epoch_loss:.4f} acc={acc:.3f}")


if __name__ == "__main__":
    path = os.environ.get("PHRASEBANK_PATH")
    if not path:
        print("Set PHRASEBANK_PATH to the dataset root, e.g.,")
        print(r"  set PHRASEBANK_PATH=F:\Business Analytics Dk\MLOps\FinancialPhraseBank-v1.0")
        print("Then run: python src\\project\\train.py")
    else:
        train_phrasebank(path, agreement="AllAgree", epochs=3, batch_size=32, lr=1e-3)
