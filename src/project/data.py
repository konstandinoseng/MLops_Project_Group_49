from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple, Union

import torch

try:
    import typer
except Exception:  # pragma: no cover
    typer = None  # type: ignore[assignment]
from torch.utils.data import Dataset


SentimentLabel = Literal["negative", "neutral", "positive"]


class FinancialPhraseBankDataset(Dataset):
    """Loader for Financial Phrase Bank v1.0.

    Reads one of the `Sentences_*Agree.txt` files under `root_path`.
    Each line: `sentence@sentiment` where sentiment ∈ {negative, neutral, positive}.
    """

    AGREEMENTS: Dict[str, str] = {
        "AllAgree": "Sentences_AllAgree.txt",
        "75Agree": "Sentences_75Agree.txt",
        "66Agree": "Sentences_66Agree.txt",
        "50Agree": "Sentences_50Agree.txt",
    }

    SENTIMENT_TO_ID: Dict[SentimentLabel, int] = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }

    def __init__(
        self,
        root_path: Union[str, Path],
        agreement: Literal["AllAgree", "75Agree", "66Agree", "50Agree"] = "AllAgree",
    ) -> None:
        self.root_path = Path(root_path)
        filename = self._agreement_file(agreement)
        file_path = self.root_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Expected dataset file not found: {file_path}")

        self.sentences: List[str] = []
        self.labels: List[int] = []
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="latin-1")
        for line in content.splitlines():
            line = line.strip()
            if not line or "@" not in line:
                continue
            sent, lab = line.rsplit("@", 1)
            lab = lab.strip().lower()
            if lab not in self.SENTIMENT_TO_ID:
                continue
            self.sentences.append(sent.strip())
            self.labels.append(self.SENTIMENT_TO_ID[lab])

        self.vocab: Dict[str, int] = {}

    @classmethod
    def _agreement_file(cls, agreement: str) -> str:
        fname = cls.AGREEMENTS.get(agreement)
        if not fname:
            raise ValueError(f"Invalid agreement '{agreement}'. Choose one of {list(cls.AGREEMENTS.keys())}")
        return fname

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index: int) -> Tuple[str, int]:
        return self.sentences[index], self.labels[index]

    def build_vocab(self, min_freq: int = 1) -> Dict[str, int]:
        freq: Dict[str, int] = {}
        for s in self.sentences:
            for tok in self.simple_tokenize(s):
                freq[tok] = freq.get(tok, 0) + 1

        vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        idx = 2
        for tok, count in sorted(freq.items()):
            if count >= min_freq:
                vocab[tok] = idx
                idx += 1
        self.vocab = vocab
        return vocab

    @staticmethod
    def simple_tokenize(text: str) -> List[str]:
        return [t for t in text.lower().split() if t]

    def encode(self, text: str) -> List[int]:
        if not self.vocab:
            self.build_vocab()
        return [self.vocab.get(tok, 1) for tok in self.simple_tokenize(text)]

    def collate_fn(self, batch: Sequence[Tuple[str, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded: List[List[int]] = [self.encode(text) for text, _ in batch]
        labels: List[int] = [lab for _, lab in batch]
        max_len = max((len(seq) for seq in encoded), default=1)
        padded = [seq + [0] * (max_len - len(seq)) for seq in encoded]
        inputs = torch.tensor(padded, dtype=torch.long)
        targets = torch.tensor(labels, dtype=torch.long)
        return inputs, targets

    def preprocess(self, output_folder: Union[str, Path], agreement: str = "AllAgree") -> None:
        out_dir = Path(output_folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not self.vocab:
            self.build_vocab()
        encoded_inputs = [self.encode(s) for s in self.sentences]
        torch.save(
            {"vocab": self.vocab, "inputs": encoded_inputs, "labels": self.labels},
            out_dir / f"phrasebank_{agreement}.pt",
        )


def preprocess(
    data_path: str,
    output_folder: str,
    agreement: Literal["AllAgree", "75Agree", "66Agree", "50Agree"] = "AllAgree",
) -> None:
    """CLI entry to preprocess Financial Phrase Bank data only.

    Requires that `data_path` contains one of the `Sentences_*Agree.txt` files.
    """
    print("Preprocessing Financial Phrase Bank...")
    data_root = Path(data_path)
    required_present = any((data_root / fname).exists() for fname in FinancialPhraseBankDataset.AGREEMENTS.values())
    if not required_present:
        raise FileNotFoundError(
            "No Financial Phrase Bank files found. Expected one of: "
            + ", ".join(FinancialPhraseBankDataset.AGREEMENTS.values())
        )
    ds = FinancialPhraseBankDataset(data_root, agreement=agreement)
    ds.preprocess(output_folder, agreement=agreement)
    print(f"Saved encoded phrasebank ({agreement}) to {Path(output_folder) / f'phrasebank_{agreement}.pt'}")


if __name__ == "__main__":
    if typer is None:
        print("Typer not installed. Install with: pip install typer")
        print("Run via: python -c \"from project.data import preprocess; preprocess('<data_path>','data/processed')\"")
    else:
        typer.run(preprocess)
