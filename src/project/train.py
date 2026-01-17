import os
from pathlib import Path
from typing import Optional, Literal

import torch
from torch.utils.data import DataLoader
try:
    import typer  # CLI
except Exception:  # pragma: no cover
    typer = None  # type: ignore[assignment]

from project.data import FinancialPhraseBankDataset
from project.model import TextSentimentModel


def train_phrasebank(
    root_path: str,
    agreement: Literal["AllAgree", "75Agree", "66Agree", "50Agree"] = "AllAgree",
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 1e-3,
    num_workers: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: Optional[int] = 2,
    save_path: Optional[str] = None,
) -> None:
    ds = FinancialPhraseBankDataset(root_path, agreement=agreement)  # e.g., F:\Business Analytics Dk\MLOps\FinancialPhraseBank-v1.0
    # Reuse cached vocab if available
    cache_file = Path("data/processed") / f"phrasebank_{agreement}.pt"
    if cache_file.exists():
        cached = torch.load(cache_file)
        vocab = cached.get("vocab") or ds.build_vocab(min_freq=1)
        ds.vocab = vocab
    else:
        vocab = ds.build_vocab(min_freq=1)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ds.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
    # Set prefetch_factor only when num_workers > 0 and a valid int is provided
    if num_workers > 0 and prefetch_factor is not None:
        loader.prefetch_factor = int(prefetch_factor)

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

    # Save model
    out_path = Path(save_path) if save_path else Path("models") / f"text_model_{agreement}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab_size": len(vocab)}, out_path)
    print(f"Saved model to {out_path}")


if typer is not None:
    app = typer.Typer(help="Training utilities for Financial Phrase Bank")

    @app.command("train")
    def train_cmd(
        path: str = typer.Option(..., "--path", help="Root path to Financial Phrase Bank"),
        agreement: Literal["AllAgree", "75Agree", "66Agree", "50Agree"] = typer.Option("AllAgree", "--agreement", help="Agreement split"),
        epochs: int = typer.Option(3, "--epochs"),
        batch_size: int = typer.Option(32, "--batch-size"),
        lr: float = typer.Option(1e-3, "--lr"),
        num_workers: int = typer.Option(2, "--num-workers"),
        pin_memory: bool = typer.Option(True, "--pin-memory"),
        persistent_workers: bool = typer.Option(True, "--persistent-workers"),
        prefetch_factor: Optional[int] = typer.Option(2, "--prefetch-factor"),
        save_path: Optional[str] = typer.Option(None, "--save-path", help="Where to save the model"),
    ):
        """Train the text sentiment model on Financial Phrase Bank."""
        train_phrasebank(
            root_path=path,
            agreement=agreement,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            save_path=save_path,
        )
else:
    app = None  # type: ignore[assignment]


if __name__ == "__main__":
    # If env var is set, allow quick start; otherwise use CLI (when typer is available)
    path = os.environ.get("PHRASEBANK_PATH")
    if path:
        train_phrasebank(path, agreement="AllAgree", epochs=3, batch_size=32, lr=1e-3)
    else:
        if typer is None:
            print("Typer not installed. Install with: pip install typer")
            print("Or set PHRASEBANK_PATH and run this file directly.")
        else:
            app()
