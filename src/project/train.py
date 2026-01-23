from pathlib import Path
from typing import Optional, Literal

import torch
from torch.utils.data import DataLoader

try:
    import typer  # CLI
except Exception:  # pragma: no cover
    typer = None  # type: ignore[assignment]

try:
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
except Exception:  # pragma: no cover
    pl = None  # type: ignore[assignment]

from project.data import FinancialPhraseBankDataset
from project.model import TextSentimentModel
from omegaconf import OmegaConf
from hydra import compose, initialize

# adding wandb
import wandb
from project.evaluate import evaluate_phrasebank

# add torch profiler


def train_phrasebank_lightning(
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
    val_split: float = 0.2,
) -> str:
    """Train using PyTorch Lightning with automatic validation and logging."""
    from project.data import FinancialPhraseBankDataModule
    from project.model import TextSentimentLightningModel

    # Setup DataModule
    datamodule = FinancialPhraseBankDataModule(
        root_path=root_path,
        agreement=agreement,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        val_split=val_split,
    )
    datamodule.setup()

    # Setup Model
    model = TextSentimentLightningModel(
        vocab_size=datamodule.get_vocab_size(),
        embedding_dim=64,
        num_classes=3,
        lr=lr,
    )

    # Setup logging and callbacks
    wandb_logger = WandbLogger(
        entity="konstandinoseng-dtu",
        project="Group_49",
        config={"epochs": epochs, "batch_size": batch_size, "learning_rate": lr, "agreement": agreement},
    )

    out_path = Path(save_path) if save_path else Path("models") / f"text_model_{agreement}.pt"
    checkpoint_callback = ModelCheckpoint(
        dirpath=out_path.parent,
        filename=out_path.stem,
        save_top_k=1,
        monitor="val/loss",
        mode="min",
    )

    # Train
    trainer = Trainer(
        max_epochs=epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=datamodule)

    # Save model in legacy format for compatibility
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.get_inner_model().state_dict(), "vocab_size": datamodule.get_vocab_size()},
        out_path,
    )
    print(f"Saved model to {out_path}")

    # Log final metrics and artifact to wandb
    artifact = wandb.Artifact(name=f"text_model_{agreement}", type="model")
    artifact.add_file(str(out_path))
    wandb.log_artifact(artifact)

    wandb.finish()
    return "Training Completed"


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
    ds = FinancialPhraseBankDataset(root_path, agreement=agreement)
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

    # Initialize wandb
    wandb.init(
        entity="konstandinoseng-dtu",
        project="Group_49",
        config={"epochs": epochs, "batch_size": batch_size, "learning_rate": lr, "agreement": agreement},
    )
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
        wandb.log({"epoch": epoch + 1, "loss": epoch_loss, "accuracy": acc})

    # Save model
    out_path = Path(save_path) if save_path else Path("models") / f"text_model_{agreement}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab_size": len(vocab)}, out_path)
    print(f"Saved model to {out_path}")
    # Evaluate the model
    eval_metrics = evaluate_phrasebank(
        root_path=root_path,
        agreement=agreement,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        model_path=str(out_path),
    )

    # Log eval metrics to the same run
    wandb.log(eval_metrics)
    wandb.summary.update(eval_metrics)

    # Store eval metrics in a single-row table
    eval_table = wandb.Table(
        columns=list(eval_metrics.keys()),
        data=[list(eval_metrics.values())],
    )
    wandb.log({"eval/final_metrics_table": eval_table})
    artifact = wandb.Artifact(name=f"text_model_{agreement}", type="model")
    artifact.add_file(str(out_path))
    wandb.log_artifact(artifact)
    return "Training Completed"


if typer is not None:
    app = typer.Typer(help="Training utilities for Financial Phrase Bank")

    @app.command("train")
    def train_cmd(
        epochs: Optional[int] = typer.Option(None),
        lr: Optional[float] = typer.Option(None),
        batch_size: Optional[int] = typer.Option(None),
        use_lightning: bool = typer.Option(True, "--lightning/--no-lightning", help="Use PyTorch Lightning trainer"),
    ):
        # FIX: Point to the root configs folder from src/project/
        with initialize(version_base=None, config_path="../../configs"):
            cfg = compose(config_name="config")

        if epochs:
            cfg.training.epochs = epochs
        if lr:
            cfg.training.lr = lr
        if batch_size:
            cfg.training.batch_size = batch_size

        print(f"Running with config:\n{OmegaConf.to_yaml(cfg)}")

        if use_lightning and pl is not None:
            print("Using PyTorch Lightning trainer...")
            train_phrasebank_lightning(
                root_path=cfg.data.root_path,
                agreement=cfg.data.agreement,
                epochs=cfg.training.epochs,
                batch_size=cfg.training.batch_size,
                lr=cfg.training.lr,
                num_workers=cfg.training.num_workers,
                pin_memory=cfg.training.pin_memory,
                persistent_workers=cfg.training.persistent_workers,
                prefetch_factor=cfg.training.prefetch_factor,
                save_path=cfg.training.save_path,
            )
        else:
            print("Using legacy trainer...")
            train_phrasebank(
                root_path=cfg.data.root_path,
                agreement=cfg.data.agreement,
                epochs=cfg.training.epochs,
                batch_size=cfg.training.batch_size,
                lr=cfg.training.lr,
                num_workers=cfg.training.num_workers,
                pin_memory=cfg.training.pin_memory,
                persistent_workers=cfg.training.persistent_workers,
                prefetch_factor=cfg.training.prefetch_factor,
                save_path=cfg.training.save_path,
            )


def main():
    """Entry point for uv run train"""
    if app:
        app()


if __name__ == "__main__":
    main()
