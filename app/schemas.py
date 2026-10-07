"""Validated request and response schemas."""

from typing import Annotated

from pydantic import BaseModel, Field

from .config import DEFAULT_MODEL, MAX_BATCH_SIZE, MAX_QUERY_LENGTH, ModelName


class InspectRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, max_length=MAX_QUERY_LENGTH)]
    model: ModelName = DEFAULT_MODEL  # type: ignore[assignment]


class BatchInspectRequest(BaseModel):
    queries: Annotated[list[str], Field(min_length=1, max_length=MAX_BATCH_SIZE)]
    model: ModelName = DEFAULT_MODEL  # type: ignore[assignment]


class Prediction(BaseModel):
    query: str
    model: str
    label: int
    is_sqli: bool
    confidence: float | None
    threat_level: str
    latency_ms: float


class BatchPrediction(BaseModel):
    predictions: list[Prediction]
    count: int
    latency_ms: float
