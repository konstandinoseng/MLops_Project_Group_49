from torch.utils.data import Dataset

from project.data import FinancialPhraseBankDataset


def test_financial_phrasebank_dataset(tmp_path):
    """Ensure FinancialPhraseBankDataset loads and is a Dataset."""
    # create a minimal AllAgree file for testing
    f = tmp_path / "Sentences_AllAgree.txt"
    f.write_text("Economy improves.@positive", encoding="utf-8")
    ds = FinancialPhraseBankDataset(tmp_path, agreement="AllAgree")
    assert isinstance(ds, Dataset)
    assert len(ds) == 1
    sent, label = ds[0]
    assert isinstance(sent, str)
    assert label in (0, 1, 2)
