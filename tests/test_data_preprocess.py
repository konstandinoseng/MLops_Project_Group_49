from project.data import preprocess, FinancialPhraseBankDataset


def test_preprocess_phrasebank(tmp_path):
    # Arrange: create minimal AllAgree file
    (tmp_path / "Sentences_AllAgree.txt").write_text(
        "Market outlook improves.@positive\nCosts remain elevated.@neutral",
        encoding="utf-8",
    )
    out_dir = tmp_path / "processed"

    # Act: preprocess to cache
    preprocess(tmp_path, out_dir, agreement="AllAgree")

    # Assert: cache file is created
    cached = out_dir / "phrasebank_AllAgree.pt"
    assert cached.exists()
    # Dataset can load back from the same root
    ds = FinancialPhraseBankDataset(tmp_path, agreement="AllAgree")
    assert len(ds) == 2
