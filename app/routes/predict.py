"""Prediction routes for single and batch requests."""

import time

from fastapi import APIRouter, HTTPException, Request

from ..config import DEFAULT_MODEL
from ..inference import predict_query
from ..schemas import BatchInspectRequest, BatchPrediction, InspectRequest, Prediction


router = APIRouter(prefix="/api/v1")


@router.get("/models")
def list_models(request: Request):
    registry = request.app.state.registry
    return {"default_model": DEFAULT_MODEL, "models": registry.available()}


@router.post("/inspect", response_model=Prediction)
def inspect_query(payload: InspectRequest, request: Request):
    try:
        return predict_query(payload.query, payload.model, request.app.state.registry)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/inspect/batch", response_model=BatchPrediction)
def inspect_batch(payload: BatchInspectRequest, request: Request):
    start = time.perf_counter()
    try:
        predictions = [
            predict_query(query, payload.model, request.app.state.registry)
            for query in payload.queries
        ]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return BatchPrediction(
        predictions=predictions,
        count=len(predictions),
        latency_ms=round((time.perf_counter() - start) * 1000, 3),
    )
