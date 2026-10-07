"""Loads and provides access to the trained model variants."""

from pathlib import Path

import joblib

from .config import DEFAULT_MODEL, MODELS, MODELS_DIR


class ModelRegistry:
    """Holds the vectorizer and every available trained classifier variant."""

    def __init__(self, models_dir: Path = MODELS_DIR, default_model: str = DEFAULT_MODEL):
        self.models_dir = Path(models_dir)
        self.default_model = default_model
        self.vectorizer = None
        self.models = {}

    def load(self):
        vectorizer_path = self.models_dir / "vectorizer.pkl"
        conditions_path = self.models_dir / "feature_conditions.pkl"
        missing = [str(path) for path in (vectorizer_path, conditions_path) if not path.is_file()]
        if missing:
            raise RuntimeError("Required model artifacts are missing: " + ", ".join(missing))

        self.vectorizer = joblib.load(vectorizer_path)
        conditions = joblib.load(conditions_path)

        for name, metadata in MODELS.items():
            path = self.models_dir / Path(metadata["path"]).name
            if not path.is_file():
                continue
            if metadata["loader"] != "joblib":
                raise RuntimeError(f"Unsupported model loader '{metadata['loader']}' for {name}")
            feature_condition = name.rsplit("_", 1)[-1]
            if feature_condition not in conditions:
                raise RuntimeError(f"Feature condition '{feature_condition}' is missing")
            self.models[name] = {
                "estimator": joblib.load(path),
                "feature_indices": conditions[feature_condition],
                "metadata": metadata,
            }

        if self.default_model not in self.models:
            raise RuntimeError(
                f"Default model '{self.default_model}' is unavailable in {self.models_dir}"
            )
        return self

    def get(self, name: str):
        try:
            return self.models[name]
        except KeyError as exc:
            raise ValueError(f"Model '{name}' is not installed") from exc

    def available(self):
        return [
            {
                "name": name,
                "classifier": name.split("_")[0],
                "features": name.rsplit("_", 1)[-1],
                "framework": self.models[name]["metadata"]["framework"],
                "description": self.models[name]["metadata"]["description"],
            }
            for name in self.models
        ]
