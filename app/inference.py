"""Feature transformation and model prediction helpers."""

import time

from .config import MAX_QUERY_LENGTH
from .registry import ModelRegistry
from .schemas import Prediction


def threat_level(probability: float) -> str:
    if probability >= 0.9:
        return "HIGH"
    if probability >= 0.6:
        return "MEDIUM"
    return "LOW"


def predict_query(query: str, model_name: str, registry: ModelRegistry) -> Prediction:
    query = query.strip()
    if not query:
        raise ValueError("Query must not be blank")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Query exceeds the {MAX_QUERY_LENGTH}-character limit")

    start = time.perf_counter()
    model_entry = registry.get(model_name)
    model = model_entry["estimator"]
    feature_indices = model_entry["feature_indices"]
    features = registry.vectorizer.transform([query])
    if feature_indices is not None:
        features = features[:, feature_indices]

    label = int(model.predict(features)[0])
    confidence = (
        float(model.predict_proba(features)[0, label])
        if hasattr(model, "predict_proba") else None
    )

    return Prediction(
        query=query,
        model=model_name,
        label=label,
        is_sqli=(label == 1),
        confidence=round(confidence, 6) if confidence is not None else None,
        threat_level=(
            threat_level(confidence if label == 1 else 1.0 - confidence)
            if confidence is not None else "UNSCORED"
        ),
        latency_ms=round((time.perf_counter() - start) * 1000, 3),
    )
