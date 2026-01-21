"""
Inference module: Load model and predict sentiment.

Pipeline:
1. Fetcher provides raw article text (from URL)
2. Retrieval extracts financial sentences
3. This module loads the model and predicts sentiment
"""

from pathlib import Path
from typing import Dict, List, Literal, Optional

import torch

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

from project.model import TextSentimentModel
from project.inference.retrieval import extract_financial_sentences


# Sentiment labels
SENTIMENT_LABELS: Dict[int, str] = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


class SentimentPredictor:
    """Load model and predict sentiment on financial sentences."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        wandb_artifact: Optional[str] = None,
        vocab_path: Optional[str] = None,
        agreement: Literal["AllAgree", "75Agree", "66Agree", "50Agree"] = "AllAgree",
    ) -> None:
        """
        Initialize predictor with model from local path or wandb.

        Args:
            model_path: Local path to model checkpoint
            wandb_artifact: W&B artifact (e.g., "entity/project/model:latest")
            vocab_path: Path to vocab file (default: data/processed/phrasebank_{agreement}.pt)
            agreement: Agreement level for default paths
        """
        self.agreement = agreement
        self.vocab: Dict[str, int] = {}
        self.model: Optional[TextSentimentModel] = None

        # Load vocab
        self._load_vocab(vocab_path)

        # Load model
        if wandb_artifact:
            self._load_from_wandb(wandb_artifact)
        elif model_path:
            self._load_from_local(model_path)
        else:
            # Default local path
            default_path = Path("models") / f"text_model_{agreement}.pt"
            if default_path.exists():
                self._load_from_local(str(default_path))
            else:
                raise ValueError("No model found. Provide model_path or wandb_artifact.")

    def _load_vocab(self, vocab_path: Optional[str] = None) -> None:
        """Load vocabulary from preprocessed data."""
        if vocab_path:
            path = Path(vocab_path)
        else:
            path = Path("data/processed") / f"phrasebank_{self.agreement}.pt"

        if path.exists():
            data = torch.load(path, weights_only=False)
            self.vocab = data.get("vocab", {})
            print(f"Loaded vocab ({len(self.vocab)} tokens) from {path}")
        else:
            raise FileNotFoundError(f"Vocab not found at {path}")

    def _load_from_wandb(self, artifact_name: str) -> None:
        """Load model from W&B artifact."""
        if wandb is None:
            raise ImportError("wandb required. Install with: pip install wandb")

        print(f"Loading model from W&B: {artifact_name}")

        # Initialize wandb run if needed
        if wandb.run is None:
            wandb.init(project="Group_49", entity="konstandinoseng-dtu", job_type="inference")

        artifact = wandb.run.use_artifact(artifact_name, type="model")
        artifact_dir = artifact.download()

        # Find model file
        model_files = list(Path(artifact_dir).glob("*.pt"))
        if not model_files:
            raise FileNotFoundError(f"No .pt file in artifact: {artifact_dir}")

        self._load_from_local(str(model_files[0]))

    def _load_from_local(self, path: str) -> None:
        """Load model from local checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        vocab_size = checkpoint.get("vocab_size", len(self.vocab))

        self.model = TextSentimentModel(vocab_size=vocab_size, embedding_dim=64, num_classes=3)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        print(f"Model loaded from {path}")

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenizer."""
        return [t for t in text.lower().split() if t]

    def _encode(self, text: str) -> List[int]:
        """Encode text to token indices."""
        return [self.vocab.get(tok, 1) for tok in self._tokenize(text)]  # 1 = <UNK>

    @torch.no_grad()
    def predict(self, sentence: str) -> Dict:
        """
        Predict sentiment for a single sentence.

        Args:
            sentence: Text to analyze

        Returns:
            Dict with sentiment, confidence, and probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        encoded = self._encode(sentence)
        if not encoded:
            encoded = [1]  # <UNK>

        inputs = torch.tensor([encoded], dtype=torch.long)
        logits = self.model(inputs)
        probs = torch.softmax(logits, dim=1).squeeze()
        pred_class = int(logits.argmax(dim=1).item())

        return {
            "text": sentence,
            "sentiment": SENTIMENT_LABELS[pred_class],
            "confidence": float(probs[pred_class].item()),
            "probabilities": {
                "negative": float(probs[0].item()),
                "neutral": float(probs[1].item()),
                "positive": float(probs[2].item()),
            },
        }

    @torch.no_grad()
    def predict_batch(self, sentences: List[str]) -> List[Dict]:
        """Predict sentiment for multiple sentences."""
        return [self.predict(s) for s in sentences]

    def analyze(self, sentences: List[str]) -> Dict:
        """
        Analyze a list of sentences and aggregate results.

        Args:
            sentences: List of financial sentences (from retrieval)

        Returns:
            Dict with overall sentiment and per-sentence predictions
        """
        if not sentences:
            return {"error": "No sentences to analyze", "sentences_analyzed": 0}

        predictions = self.predict_batch(sentences)

        # Aggregate
        counts = {"negative": 0, "neutral": 0, "positive": 0}
        for pred in predictions:
            counts[pred["sentiment"]] += 1

        overall = max(counts, key=counts.get)  # type: ignore

        return {
            "sentences_analyzed": len(sentences),
            "overall_sentiment": overall,
            "sentiment_distribution": counts,
            "predictions": predictions,
        }


def run_inference(
    text: str,
    model_path: Optional[str] = None,
    wandb_artifact: Optional[str] = None,
) -> Dict:
    """
    Full pipeline: retrieval + prediction.

    Args:
        text: Raw article text (from fetcher)
        model_path: Local model path
        wandb_artifact: W&B artifact name

    Returns:
        Analysis results
    """
    # Step 1: Retrieval - extract financial sentences
    sentences = extract_financial_sentences(text)

    # Step 2: Load model and predict
    predictor = SentimentPredictor(
        model_path=model_path,
        wandb_artifact=wandb_artifact,
    )

    # Step 3: Analyze
    return predictor.analyze(sentences)


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Apple Inc. reported quarterly revenue of $89.5 billion, an increase of 8% year over year.
    The weather was nice today. iPhone sales exceeded analyst expectations, driving strong growth.
    The company's gross margin improved to 43.3%, reflecting operational efficiency.
    Investors remain optimistic about the company's future prospects.
    The stock surged 5% in after-hours trading following the earnings beat.
    """

    # Run with local model
    result = run_inference(sample_text, model_path="models/text_model_AllAgree.pt")
    print(f"Overall sentiment: {result['overall_sentiment']}")
    print(f"Distribution: {result['sentiment_distribution']}")
