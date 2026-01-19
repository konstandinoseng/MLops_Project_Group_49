from fastapi import FastAPI
from http import HTTPStatus
from omegaconf import OmegaConf
from hydra import compose, initialize
from .train import train_phrasebank
from .evaluate import evaluate_phrasebank
from pydantic import BaseModel

app = FastAPI()


class TrainParams(BaseModel):
    epochs: int
    batch_size: int
    lr: float


@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


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


# @app.get("/evaluate")
# def evaluate_backend():

#     evaluate_phrasebank(
#         root_path=path,
#         agreement=agreement,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         persistent_workers=persistent_workers,
#         model_path=model_path,
#     )

#     return "evaluate"
