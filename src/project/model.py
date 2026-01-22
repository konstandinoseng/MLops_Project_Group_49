from torch import nn
import torch

try:
    import pytorch_lightning as pl
    from torchmetrics import Accuracy, Precision, Recall, F1Score
except Exception:  # pragma: no cover
    pl = None  # type: ignore[assignment]


class TextSentimentModel(nn.Module):
    """Bag-of-words style text classifier using average embeddings.

    Inputs are token indices of shape [B, T]. Output logits shape [B, num_classes].
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 64, num_classes: int = 3) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        emb = self.embedding(x)  # [B, T, D]
        # Create mask for non-pad tokens
        mask = (x != 0).float()  # [B, T]
        # Avoid division by zero
        lengths = mask.sum(dim=1).clamp(min=1.0)  # [B]
        # Sum embeddings over time and average
        summed = (emb * mask.unsqueeze(-1)).sum(dim=1)  # [B, D]
        avg = summed / lengths.unsqueeze(-1)  # [B, D]
        logits = self.fc(avg)  # [B, C]
        return logits


# ============================================================================
# PyTorch Lightning Module
# ============================================================================

if pl is not None:

    class TextSentimentLightningModel(pl.LightningModule):
        """Lightning wrapper for TextSentimentModel with automatic metrics."""

        def __init__(
            self,
            vocab_size: int,
            embedding_dim: int = 64,
            num_classes: int = 3,
            lr: float = 1e-3,
        ) -> None:
            super().__init__()
            self.save_hyperparameters()
            self.model = TextSentimentModel(vocab_size, embedding_dim, num_classes)
            self.criterion = nn.CrossEntropyLoss()
            self.lr = lr

            # Metrics
            self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
            self.val_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
            self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

        def training_step(self, batch, batch_idx):
            inputs, targets = batch
            logits = self(inputs)
            loss = self.criterion(logits, targets)
            preds = logits.argmax(dim=1)

            self.train_acc(preds, targets)
            self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            inputs, targets = batch
            logits = self(inputs)
            loss = self.criterion(logits, targets)
            preds = logits.argmax(dim=1)

            self.val_acc(preds, targets)
            self.val_precision(preds, targets)
            self.val_recall(preds, targets)
            self.val_f1(preds, targets)

            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/precision", self.val_precision, on_step=False, on_epoch=True)
            self.log("val/recall", self.val_recall, on_step=False, on_epoch=True)
            self.log("val/f1", self.val_f1, on_step=False, on_epoch=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.lr)

        def get_inner_model(self) -> TextSentimentModel:
            """Return the inner TextSentimentModel for saving/loading."""
            return self.model


if __name__ == "__main__":
    # Text model sanity check
    text_model = TextSentimentModel(vocab_size=100)
    x_tokens = torch.randint(0, 99, (4, 10))
    print(f"Text model output shape: {text_model(x_tokens).shape}")
