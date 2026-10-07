"""Health and readiness routes."""

from fastapi import APIRouter, HTTPException, Request

from ..config import DEFAULT_MODEL


router = APIRouter()


@router.get("/health")
def readiness(request: Request):
    registry = getattr(request.app.state, "registry", None)
    if registry is None or not registry.models:
        raise HTTPException(status_code=503, detail="Models are not loaded")
    return {
        "status": "ready",
        "default_model": DEFAULT_MODEL,
        "available_models": [model["name"] for model in registry.available()],
    }
