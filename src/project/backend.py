from fastapi import FastAPI, HTTPException
from http import HTTPStatus
from omegaconf import OmegaConf
from hydra import compose, initialize
from .train import train_phrasebank
from .evaluate import evaluate_phrasebank
from .model import TextSentimentModel
from .data import FinancialPhraseBankDataset
from pydantic import BaseModel
import torch
from pathlib import Path
from contextlib import asynccontextmanager

# Global model variable
model_artifact = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    model_path = Path("models/text_model_AllAgree.pt")
    if model_path.exists():
        checkpoint = torch.load(model_path)
        # We need the vocab to initialize the model correctly
        # In a real scenario, you'd save vocab size or the vocab itself with the model
        # Here we assume a default or infer from checkpoint if available
        vocab_size = checkpoint.get("vocab_size", 20000) # Fallback if not saved
        
        model = TextSentimentModel(vocab_size=vocab_size)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model_artifact["model"] = model
        
        # Load vocab for tokenization
        # For simplicity, we might need to rebuild or load it. 
        # Ideally, save the vocab dictionary in the checkpoint.
        # Here we will just load the processed data if available to get the vocab
        # Or simpler: assume standard vocab if saved.
        # Let's assume the checkpoint has what we need or we can't fully predict without it.
        # Checkpoint in train.py saves: {"state_dict": ..., "vocab_size": ...}
        # It DOES NOT save the actual vocab dict, which is a limitation for inference.
        # FIX: We need the vocab to tokenize input strings. 
    else:
        print("Warning: Model file not found. Prediction endpoint will fail.")
    
    yield
    # Clean up on shutdown
    model_artifact.clear()

app = FastAPI(lifespan=lifespan)


class TrainParams(BaseModel):
    epochs: int
    batch_size: int
    lr: float

class PredictRequest(BaseModel):
    text: str

@app.get("/")
def root():
    """Health check."""
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }

@app.post("/predict")
def predict(request: PredictRequest):
    if "model" not in model_artifact:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # LIMITATION: We don't have the vocab from the training process loaded here efficiently.
    # For a robust implementation, we should load the vocab from `data/processed/phrasebank_AllAgree.pt`
    # or ensure train.py saves the vocab dict.
    
    # Temporary fix: Load vocab from processed data file if it exists
    processed_path = Path("data/processed/phrasebank_AllAgree.pt")
    if not processed_path.exists():
         raise HTTPException(status_code=500, detail="Vocab not found")
         
    data = torch.load(processed_path)
    vocab = data["vocab"]
    
    # Tokenize
    tokens = [vocab.get(t, 1) for t in request.text.lower().split() if t]
    if not tokens: # Handle empty or unknown words
        return {"sentiment": "neutral", "confidence": 0.0}
        
    input_tensor = torch.tensor([tokens], dtype=torch.long)
    
    with torch.no_grad():
        logits = model_artifact["model"](input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        
    # Mapping from data.py
    id_to_sentiment = {0: "negative", 1: "neutral", 2: "positive"}
    
    return {
        "sentiment": id_to_sentiment.get(pred_idx, "unknown"),
        "confidence": probs[0][pred_idx].item()
    }


@app.post("/train")
def train_backend(params: TrainParams):
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="config")

    result = train_phrasebank(
        root_path=cfg.data.root_path,
        agreement=cfg.data.agreement,
        epochs=params.epochs,
        batch_size=params.batch_size,
        lr=params.lr,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        persistent_workers=cfg.training.persistent_workers,
        prefetch_factor=cfg.training.prefetch_factor,
        save_path=cfg.training.save_path,
    )
    return {
        "status": "success",
        "message": result,
        "epochs": params.epochs,
        "batch_size": params.batch_size,
        "lr": params.lr,
    }
