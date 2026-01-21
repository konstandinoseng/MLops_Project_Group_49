"""
Retrieval module: Extract relevant financial sentences from text.

Filters raw article text to keep only sentences containing financial keywords.
"""

import re
from typing import List, Set


# Financial keywords for filtering
FINANCIAL_KEYWORDS: Set[str] = {
    # Performance indicators
    "profit",
    "loss",
    "revenue",
    "earnings",
    "growth",
    "decline",
    "increase",
    "decrease",
    "rise",
    "fall",
    "gain",
    "drop",
    "surge",
    "plunge",
    "soar",
    "jump",
    "slump",
    "recover",
    "rebound",
    "slide",
    "tumble",
    # Financial terms
    "stock",
    "share",
    "shares",
    "market",
    "investor",
    "investors",
    "dividend",
    "quarter",
    "quarterly",
    "annual",
    "fiscal",
    "year",
    "forecast",
    "outlook",
    "guidance",
    "target",
    "estimate",
    "estimates",
    "expect",
    "expects",
    "expected",
    "projection",
    "projections",
    "results",
    "performance",
    "beat",
    "miss",
    "exceeded",
    # Sentiment indicators
    "positive",
    "negative",
    "strong",
    "weak",
    "bullish",
    "bearish",
    "robust",
    "optimistic",
    "pessimistic",
    "confident",
    "concerned",
    "worried",
    "cautious",
    "upbeat",
    "downbeat",
    "promising",
    "disappointing",
    "impressive",
    "lackluster",
    # Actions
    "buy",
    "sell",
    "hold",
    "upgrade",
    "downgrade",
    "recommend",
    "rating",
    "outperform",
    "underperform",
    "overweight",
    "underweight",
    # Metrics
    "eps",
    "ebitda",
    "margin",
    "margins",
    "roi",
    "roe",
    "p/e",
    "ratio",
    "billion",
    "million",
    "percent",
    "percentage",
    "%",
    "basis",
    "points",
    # Company actions
    "acquisition",
    "merger",
    "deal",
    "buyback",
    "restructuring",
    "layoffs",
    "expansion",
    "investment",
    "ipo",
    "offering",
    "debt",
    "loan",
    "credit",
}


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    Args:
        text: Raw text to split

    Returns:
        List of sentence strings
    """
    pattern = r"(?<=[.!?])\s+(?=[A-Z])"
    return re.split(pattern, text)


def is_financial(sentence: str, keywords: Set[str] = FINANCIAL_KEYWORDS) -> bool:
    """
    Check if a sentence contains financial keywords.

    Args:
        sentence: Text to check
        keywords: Set of financial keywords

    Returns:
        True if sentence contains at least one financial keyword
    """
    words = set(sentence.lower().split())
    return bool(words & keywords)


def extract_financial_sentences(
    text: str,
    min_length: int = 20,
    max_length: int = 500,
) -> List[str]:
    """
    Extract all sentences containing financial keywords.

    Args:
        text: Raw article text
        min_length: Minimum sentence length (characters)
        max_length: Maximum sentence length (characters)

    Returns:
        List of financial sentences
    """
    sentences = split_sentences(text)

    result: List[str] = []
    for sentence in sentences:
        sentence = sentence.strip()

        # Filter by length
        if len(sentence) < min_length or len(sentence) > max_length:
            continue

        # Keep if contains financial keywords
        if is_financial(sentence):
            result.append(sentence)

    return result


if __name__ == "__main__":
    sample_text = """
    Apple Inc. reported quarterly revenue of $89.5 billion, an increase of 8% year over year.
    The weather was nice today. iPhone sales exceeded analyst expectations, driving strong growth.
    The company's gross margin improved to 43.3%, reflecting operational efficiency.
    Investors remain optimistic about the company's future prospects.
    CEO Tim Cook expressed confidence in the upcoming product lineup.
    The stock surged 5% in after-hours trading following the earnings beat.
    """

    results = extract_financial_sentences(sample_text)
    print("Extracted financial sentences:")
    for i, sentence in enumerate(results, 1):
        print(f"{i}. {sentence}")
