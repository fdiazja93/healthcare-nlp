"""FastAPI service: pulls the model from S3 on startup and serves /predict with JSON logs."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import boto3
from fastapi import FastAPI, Request
from pythonjsonlogger import jsonlogger

from healthcare_nlp.api.inference import load_model, infer
from healthcare_nlp.api.schemas import PredictionRequest, PredictionResponse

# JSON logging setup
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    fmt="%(asctime)s %(name)s %(levelname)s %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

_LOCAL_PATH_TO_MODEL = Path("/tmp/model")
_BUCKET = "healthcare-nlp-853301924030"
_PREFIX = "api/model"
_MODEL_NAME = "model_v0"


def download_model_from_s3(
    bucket: str = _BUCKET,
    prefix: str = _PREFIX,
    local_path: Path = _LOCAL_PATH_TO_MODEL,
) -> None:
    """Sync the serialized model and tokenizer from S3 into the container's local path."""
    s3 = boto3.client("s3")
    local_path.mkdir(parents=True, exist_ok=True)
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    for obj in response["Contents"]:
        key = obj["Key"]
        filename = Path(key).name
        s3.download_file(bucket, key, str(local_path / filename))


@asynccontextmanager
async def lifespan(app: FastAPI):
    download_model_from_s3()
    load_model(_LOCAL_PATH_TO_MODEL)
    yield


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    log_data: dict = {
        "endpoint": request.url.path,
        "method": request.method,
        "latency_ms": round(latency * 1000, 2),
        "status_code": response.status_code,
    }

    if hasattr(request.state, "input_text_length"):
        log_data["input_text_length"] = request.state.input_text_length
    if hasattr(request.state, "predicted_label"):
        log_data["predicted_label"] = request.state.predicted_label
    if hasattr(request.state, "probability"):
        log_data["probability"] = request.state.probability

    logger.info("request", extra=log_data)
    return response


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(pred_req: PredictionRequest, request: Request) -> PredictionResponse:
    result = infer(pred_req.input_text, model_name=_MODEL_NAME)

    request.state.input_text_length = len(pred_req.input_text)
    request.state.predicted_label = result.label
    request.state.probability = result.probability

    return result
