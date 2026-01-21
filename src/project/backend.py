from fastapi import FastAPI
from http import HTTPStatus
from .inference.inference import run_inference

app = FastAPI()


class InferenceParams:
    url: str


@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.post("/inference")
def inference_backend(params: InferenceParams):
    message = run_inference(
        url=params, wandb_artifact="konstandinoseng-dtu/Project-src_project/text_model_AllAgree:v87"
    )

    return message
