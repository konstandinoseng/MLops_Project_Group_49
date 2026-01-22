#################################################################################
# With pytorch lightning evaluate is run from validation in model.py therefore  #
# this script is deprecated I have kept it here for legacy reasons              #
#################################################################################

from pathlib import Path
from typing import Optional, Tuple, Literal
import click
import torch

try:
    import typer
except Exception:  # pragma: no cover
    typer = None  # type: ignore[assignment]
from torch.utils.data import DataLoader

from project.data import FinancialPhraseBankDataset
from project.model import TextSentimentModel

from project.profiling import config_from_env, torch_profile


# run profiling with:
# $env:TORCH_PROFILER="1"; uv run python -m project.evaluate --path "data\raw" --agreement "AllAgree"
# uv run tensorboard --logdir=./log
# open http://localhost:6006/ in browser
def _metrics(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float, float, float]:
    """Return accuracy, precision (macro), recall (macro), f1 (macro)."""
    num_classes = int(targets.max().item()) + 1 if targets.numel() > 0 else 3
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for p, t in zip(preds.view(-1), targets.view(-1)):
        conf[int(t), int(p)] += 1
    # per-class metrics
    precs = []
    recs = []
    f1s = []
    for c in range(num_classes):
        tp = conf[c, c].item()
        fp = conf[:, c].sum().item() - tp
        fn = conf[c, :].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precs.append(precision)
        recs.append(recall)
        f1s.append(f1)
    accuracy = conf.diag().sum().item() / max(conf.sum().item(), 1)
    precision_macro = sum(precs) / len(precs)
    recall_macro = sum(recs) / len(recs)
    f1_macro = sum(f1s) / len(f1s)
    return accuracy, precision_macro, recall_macro, f1_macro


def evaluate_phrasebank(
    root_path: str,
    agreement: Literal["AllAgree", "75Agree", "66Agree", "50Agree"] = "AllAgree",
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    model_path: Optional[str] = None,
) -> None:
    """Evaluate saved model on Financial Phrase Bank and print metrics."""
    ds = FinancialPhraseBankDataset(root_path, agreement=agreement)
    # Load vocab from cache if present for consistent encoding
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
        shuffle=False,
        collate_fn=ds.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    # Load model
    model_file = Path(model_path) if model_path else Path("models") / f"text_model_{agreement}.pt"
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found at {model_file}. Train or provide --model-path.")
    ckpt = torch.load(model_file)
    model = TextSentimentModel(vocab_size=len(vocab), embedding_dim=64, num_classes=3)
    model.load_state_dict(ckpt["state_dict"])  # type: ignore[index]
    model.eval()

    all_preds = []
    all_targets = []
    cfg = config_from_env(default_run_name=f"phrasebank_eval_{agreement}", steps=30)
    # Evaluation loop with profiling
    with torch.no_grad():
        with torch_profile(cfg) as prof:
            for step, (inputs, targets) in enumerate(loader):
                logits = model(inputs)
                preds = logits.argmax(dim=1)
                all_preds.append(preds)
                all_targets.append(targets)

                # profiler iteration boundary
                if prof is not None:
                    prof.step()

                # cap profiling length (strongly recommended)
                if prof is not None and step + 1 >= cfg.steps:
                    break
    preds_t = torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long)
    targets_t = torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long)

    acc, prec, rec, f1 = _metrics(preds_t, targets_t)
    metrics = {
        "eval/accuracy": acc,
        "eval/precision_macro": prec,
        "eval/recall_macro": rec,
        "eval/f1_macro": f1,
    }
    print(f"accuracy={acc:.3f} precision_macro={prec:.3f} " f"recall_macro={rec:.3f} f1_macro={f1:.3f}")
    return metrics


if typer is not None:
    app = typer.Typer(help="Evaluation utilities for Financial Phrase Bank")

    @app.command()
    def eval_cmd(
        path: str = typer.Option(..., "--path", help="Root path to Financial Phrase Bank"),
        agreement: str = typer.Option(
            "AllAgree",
            "--agreement",
            click_type=click.Choice(["AllAgree", "75Agree", "66Agree", "50Agree"]),
        ),
        batch_size: int = typer.Option(64, "--batch-size"),
        num_workers: int = typer.Option(2, "--num-workers"),
        pin_memory: bool = typer.Option(True, "--pin-memory/--no-pin-memory"),
        persistent_workers: bool = typer.Option(True, "--persistent-workers/--no-persistent-workers"),
        model_path: Optional[str] = typer.Option(None, "--model-path", help="Path to saved model"),
    ):
        evaluate_phrasebank(
            root_path=path,
            agreement=agreement,  # type: ignore[arg-type]
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            model_path=model_path,
        )
else:
    app = None  # type: ignore[assignment]


if __name__ == "__main__":
    if typer is None or app is None:
        print("Typer not installed. Install with: pip install typer")
    else:
        app()
