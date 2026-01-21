from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from http import HTTPStatus
from .inference.inference import run_inference

from contextlib import asynccontextmanager
import torch
from pathlib import Path
from .model import TextSentimentModel

# Global model variable
model_artifact = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    # Use the specific model file we know exists
    model_path = Path("models/text_model_AllAgree.pt")
    if model_path.exists():
        checkpoint = torch.load(model_path)
        # Fallback if vocab_size not in checkpoint
        vocab_size = checkpoint.get("vocab_size", 20000) 
        
        model = TextSentimentModel(vocab_size=vocab_size)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model_artifact["model"] = model
        print(f"Model loaded from {model_path}")
    else:
        print("Warning: Model file not found. Prediction endpoint will fail.")
    
    yield
    # Clean up on shutdown
    model_artifact.clear()

app = FastAPI(
    title="Financial Sentiment API",
    description="Analyze financial article sentiment using AI",
    version="1.0.0",
    lifespan=lifespan
)


# Request Models
class InferenceRequest(BaseModel):
    """Request model for sentiment analysis."""

    url: HttpUrl
    """URL of the financial article to analyze"""

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
            wandb_artifact=params.wandb_artifact,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")
