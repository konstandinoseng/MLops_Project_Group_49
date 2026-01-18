from torch.utils.data import DataLoader

from project.data import FinancialPhraseBankDataset
from project.model import TextSentimentModel


def test_text_sentiment_model_output_shape(tmp_path):
    # Prepare a tiny dataset file
    (tmp_path / "Sentences_AllAgree.txt").write_text(
        "Stock rises strongly.@positive\nRevenue remained flat.@neutral\nLosses widened significantly.@negative",
        encoding="utf-8",
    )
    ds = FinancialPhraseBankDataset(tmp_path, agreement="AllAgree")
    vocab = ds.build_vocab(min_freq=1)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=ds.collate_fn)

    model = TextSentimentModel(vocab_size=len(vocab), embedding_dim=32, num_classes=3)
    for inputs, targets in loader:
        # inputs: [B, T], targets: [B]
        logits = model(inputs)
        assert logits.shape[0] == inputs.shape[0]
        assert logits.shape[1] == 3  # 3 classes
        assert targets.ndim == 1


def test_collate_pads_sequences(tmp_path):
    (tmp_path / "Sentences_AllAgree.txt").write_text(
        "Short.@neutral\nA much longer sentence appears here.@positive",
        encoding="utf-8",
    )
    ds = FinancialPhraseBankDataset(tmp_path, agreement="AllAgree")
    ds.build_vocab(min_freq=1)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=ds.collate_fn)
    inputs, targets = next(iter(loader))
    # inputs: [B, T] where T is max sequence length in batch
    assert inputs.ndim == 2
    assert inputs.shape[0] == 2
    assert targets.shape[0] == 2
