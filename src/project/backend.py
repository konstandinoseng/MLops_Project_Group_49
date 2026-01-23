import json
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from http import HTTPStatus
from pydantic import BaseModel, HttpUrl

if TYPE_CHECKING:
    from google.cloud import storage

try:
    from google.cloud import storage
except ImportError:  # pragma: no cover - optional dependency for local runs
    storage = None  # type: ignore[assignment]

from .inference.inference import run_inference
from .inference.scraper import source_from_url

app = FastAPI(
    title="Financial Sentiment API",
    description="Analyze financial article sentiment using AI",
    version="1.0.0",
)

# GCS logging configuration (used later for drift logging)
GCS_BUCKET = os.getenv("GCS_BUCKET")
GCS_PREFIX = os.getenv("GCS_PREFIX", "drift-logs")


def _gcs_client() -> "storage.Client":  #type: ignore[name-defined]
    """Create a GCS client (requires google-cloud-storage)."""
    if storage is None:
        raise RuntimeError("google-cloud-storage is required for GCS logging")
    return storage.Client()


def _log_json_to_gcs(payload: dict) -> None:
    """Write one JSON payload to GCS as a single object."""
    if not GCS_BUCKET:
        raise RuntimeError("GCS_BUCKET is not set")

    client = _gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    object_name = f"{GCS_PREFIX}/inference/{timestamp}.json"
    blob = bucket.blob(object_name)
    blob.upload_from_string(json.dumps(payload), content_type="application/json")




# Request Models
class InferenceRequest(BaseModel):
    """Request model for sentiment analysis."""

    url: HttpUrl
    """URL of the financial article to analyze"""

    model_path: str = "models/text_model_AllAgree.pt"

    wandb_artifact: str = "konstandinoseng-dtu/Group_49/text_model_AllAgree:latest"
    """W&B model artifact to use (optional)"""


class AnalysisResponse(BaseModel):
    """Response model for sentiment analysis."""

    url: str
    sentences_analyzed: int
    overall_sentiment: str
    sentiment_distribution: dict
    predictions: list


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "message": "Financial Sentiment API",
        "status": HTTPStatus.OK.phrase,
        "status_code": HTTPStatus.OK.value,
    }


@app.get("/health")
def health_check():
    """API health status."""
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_sentiment(params: InferenceRequest):
    """
    Analyze sentiment of a financial article.

    Args:
        params: InferenceRequest with URL and optional wandb_artifact

    Returns:
        AnalysisResponse with sentiment analysis results
    """
    try:
        # Convert HttpUrl to string (includes full https://)
        article_url = str(params.url)

        result = run_inference(
            url=article_url,  # Pass full URL with https://
            model_path=params.model_path,
        )

        # Add URL to response
        result["url"] = article_url

        # Step 2: log one row per request to GCS (if configured)
        if GCS_BUCKET:
            sentence_texts = [
                pred.get("text")
                for pred in result.get("predictions", [])
                if isinstance(pred, dict) and pred.get("text")
            ]
            log_payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "text": " ".join(sentence_texts) if sentence_texts else None,
                "prediction": result.get("overall_sentiment"),
                "proba": None,
                "language": None,
                "source": source_from_url(article_url),
                "model_version": None,
                "preprocessing_version": None,
                "url": article_url,
            }
            _log_json_to_gcs(log_payload)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")
