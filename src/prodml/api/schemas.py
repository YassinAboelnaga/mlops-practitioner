from pydantic import BaseModel, Field
from typing import List


class PredictionRequest(BaseModel):
    PU_DO: str
    trip_distance: float = Field(gt=0, lt=200)

    model_config = {
        "json_schema_extra": {"example": {"PU_DO": "74_42", "trip_distance": 3.2}}
    }


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    correlation_id: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    items: List[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
