import time
import uuid
import hashlib
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from prodml.logging_conf import setup_logging
from prodml.context import set_correlation_id, get_correlation_id
from prodml.predict import DurationPredictor
from prodml.config import settings
from prodml.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)

setup_logging()
logger = logging.getLogger("prodml.api")

ml_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_state["predictor"] = DurationPredictor.load(settings.model_path)
    with open(settings.model_path, "rb") as f:
        ml_state["artifact_hash"] = hashlib.md5(f.read()).hexdigest()
    yield
    ml_state.clear()


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = correlation_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "Invalid input", "details": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    logger.error(f"unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


@app.get("/health")
async def health():
    if "predictor" not in ml_state:
        return JSONResponse(status_code=503, content={"status": "model not loaded"})
    return {"status": "ok"}


@app.get("/metadata")
async def metadata():
    return {
        "model_version": "0.1.0",
        "framework": "scikit-learn",
        "feature_names": ["PU_DO", "trip_distance"],
        "artifact_hash": ml_state.get("artifact_hash"),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    start = time.perf_counter()
    prediction = ml_state["predictor"].predict_one(req.model_dump())
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(f"prediction served: {prediction:.2f} min")
    return PredictionResponse(
        prediction=prediction,
        model_version="0.1.0",
        correlation_id=get_correlation_id(),
        latency_ms=latency_ms,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(req: BatchPredictionRequest):
    results = []
    for item in req.items:
        start = time.perf_counter()
        pred = ml_state["predictor"].predict_one(item.model_dump())
        latency_ms = (time.perf_counter() - start) * 1000
        results.append(
            PredictionResponse(
                prediction=pred,
                model_version="0.1.0",
                correlation_id=get_correlation_id(),
                latency_ms=latency_ms,
            )
        )
    return BatchPredictionResponse(predictions=results)
