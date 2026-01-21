"""
Inference package for financial sentiment analysis.

Modules:
- retrieval: Extract relevant financial sentences from text
- fetcher: Fetch article content from URLs (to be implemented)
- inference: Load model and predict sentiment
"""

from project.inference.retrieval import (
    extract_financial_sentences,
    is_financial,
    split_sentences,
    FINANCIAL_KEYWORDS,
)

from project.inference.inference import (
    SentimentPredictor,
    run_inference,
    SENTIMENT_LABELS,
)

__all__ = [
    # Retrieval
    "extract_financial_sentences",
    "is_financial",
    "split_sentences",
    "FINANCIAL_KEYWORDS",
    # Inference
    "SentimentPredictor",
    "run_inference",
    "SENTIMENT_LABELS",
]
