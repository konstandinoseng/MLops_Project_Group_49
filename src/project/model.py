from torch import nn
import torch


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


if __name__ == "__main__":
    # Text model sanity check
    text_model = TextSentimentModel(vocab_size=100)
    x_tokens = torch.randint(0, 99, (4, 10))
    print(f"Text model output shape: {text_model(x_tokens).shape}")
