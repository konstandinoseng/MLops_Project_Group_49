from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch

try:
    import typer
except Exception:  # pragma: no cover
    typer = None  # type: ignore[assignment]

try:
    import pytorch_lightning as pl
except Exception:  # pragma: no cover
    pl = None  # type: ignore[assignment]

from torch.utils.data import Dataset, DataLoader, random_split


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
    agreement: str = typer.Option(
        "AllAgree",
        "--agreement",
        "-a",
        help="Agreement level: AllAgree, 75Agree, 66Agree, or 50Agree",
    ),
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


# ============================================================================
# PyTorch Lightning DataModule
# ============================================================================

if pl is not None:

    class FinancialPhraseBankDataModule(pl.LightningDataModule):
        """Lightning DataModule for Financial Phrase Bank dataset.

        Handles train/val splitting and dataloader creation automatically.
        """

        def __init__(
            self,
            root_path: Union[str, Path],
            agreement: Literal["AllAgree", "75Agree", "66Agree", "50Agree"] = "AllAgree",
            batch_size: int = 32,
            num_workers: int = 2,
            pin_memory: bool = True,
            persistent_workers: bool = True,
            val_split: float = 0.2,
        ) -> None:
            super().__init__()
            self.root_path = Path(root_path)
            self.agreement = agreement
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.pin_memory = pin_memory
            self.persistent_workers = persistent_workers and num_workers > 0
            self.val_split = val_split

            self.train_dataset: Optional[Dataset] = None
            self.val_dataset: Optional[Dataset] = None
            self.vocab: Dict[str, int] = {}

        def setup(self, stage: Optional[str] = None) -> None:
            """Load dataset and create train/val split."""
            full_dataset = FinancialPhraseBankDataset(self.root_path, agreement=self.agreement)

            # Load vocab from cache if available
            cache_file = Path("data/processed") / f"phrasebank_{self.agreement}.pt"
            if cache_file.exists():
                cached = torch.load(cache_file)
                self.vocab = cached.get("vocab") or full_dataset.build_vocab(min_freq=1)
                full_dataset.vocab = self.vocab
            else:
                self.vocab = full_dataset.build_vocab(min_freq=1)

            # Split into train/val
            total_size = len(full_dataset)
            val_size = int(total_size * self.val_split)
            train_size = total_size - val_size

            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

            # Store reference to full dataset for collate_fn
            self._full_dataset = full_dataset

        def train_dataloader(self) -> DataLoader:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self._full_dataset.collate_fn,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )

        def val_dataloader(self) -> DataLoader:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self._full_dataset.collate_fn,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )

        def get_vocab_size(self) -> int:
            """Return vocabulary size for model initialization."""
            return len(self.vocab)


if __name__ == "__main__":
    if typer is None:
        print("Typer not installed. Install with: pip install typer")
        print("Run via: python -c \"from project.data import preprocess; preprocess('<data_path>','data/processed')\"")
    else:
        typer.run(preprocess)
