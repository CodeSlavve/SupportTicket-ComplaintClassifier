import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from src.api.logger import logger
from src.api.model import ModelService
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)

BASE_DIR = Path(__file__).resolve().parents[2]

PIPELINE_PATH = BASE_DIR / "models" / "xgb_pipeline.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"


model_service = ModelService(
    pipeline_path=PIPELINE_PATH,
    encoder_path=ENCODER_PATH
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artifacts...")
    
    model_service.load()
    try:
        model_service.load()
        logger.info("Model artifacts loaded successfully")
    except Exception:
        logger.exception("Failed to load model artifacts")
        raise

    yield

    logger.info("API shutdown")


app = FastAPI(
    title="CFPB Complaint Classifier",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start_time = time.perf_counter()

    category, confidence = model_service.predict(request.text)

    latency_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "prediction | endpoint=/predict | input_length=%d | "
        "category=%s | confidence=%.4f | latency_ms=%.2f",
        len(request.text),
        category,
        confidence,
        latency_ms,
    )

    return PredictionResponse(
        category=category,
        confidence=confidence,
    )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    start_time = time.perf_counter()

    results = model_service.predict_batch(request.text)

    latency_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "batch_prediction | endpoint=/predict/batch | "
        "batch_size=%d | latency_ms=%.2f",
        len(request.text),
        latency_ms,
    )

    return BatchPredictionResponse(
        predictions=results
    )