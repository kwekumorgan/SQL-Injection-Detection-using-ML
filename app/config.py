"""Configuration for the SQL injection detection API."""

import os
from pathlib import Path
from typing import Literal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(PROJECT_ROOT / "models")))
DEFAULT_MODEL = os.getenv("SQLI_MODEL", "lr_chi2")
MAX_QUERY_LENGTH = 10_000
MAX_BATCH_SIZE = 100

MODELS = {
    "nb_baseline": {
        "path": MODELS_DIR / "nb_baseline.pkl",
        "loader": "joblib",
        "framework": "scikit-learn",
        "description": "Multinomial Naive Bayes using all TF-IDF features",
    },
    "nb_chi2": {
        "path": MODELS_DIR / "nb_chi2.pkl",
        "loader": "joblib",
        "framework": "scikit-learn",
        "description": "Multinomial Naive Bayes using chi-square selected features",
    },
    "svm_baseline": {
        "path": MODELS_DIR / "svm_baseline.pkl",
        "loader": "joblib",
        "framework": "scikit-learn",
        "description": "Linear SVM using all TF-IDF features",
    },
    "svm_chi2": {
        "path": MODELS_DIR / "svm_chi2.pkl",
        "loader": "joblib",
        "framework": "scikit-learn",
        "description": "Linear SVM using chi-square selected features",
    },
    "lr_baseline": {
        "path": MODELS_DIR / "lr_baseline.pkl",
        "loader": "joblib",
        "framework": "scikit-learn",
        "description": "Logistic regression using all TF-IDF features",
    },
    "lr_chi2": {
        "path": MODELS_DIR / "lr_chi2.pkl",
        "loader": "joblib",
        "framework": "scikit-learn",
        "description": "Logistic regression using chi-square selected features",
    },
    "dt_baseline": {
        "path": MODELS_DIR / "dt_baseline.pkl",
        "loader": "joblib",
        "framework": "scikit-learn",
        "description": "Decision tree using all TF-IDF features",
    },
    "dt_chi2": {
        "path": MODELS_DIR / "dt_chi2.pkl",
        "loader": "joblib",
        "framework": "scikit-learn",
        "description": "Decision tree using chi-square selected features",
    },
    "knn_baseline": {
        "path": MODELS_DIR / "knn_baseline.pkl",
        "loader": "joblib",
        "framework": "scikit-learn",
        "description": "K-nearest neighbors using all TF-IDF features",
    },
    "knn_chi2": {
        "path": MODELS_DIR / "knn_chi2.pkl",
        "loader": "joblib",
        "framework": "scikit-learn",
        "description": "K-nearest neighbors using chi-square selected features",
    },
}

MODEL_NAMES = tuple(MODELS)
ModelName = Literal[
    "nb_baseline", "nb_chi2",
    "svm_baseline", "svm_chi2",
    "lr_baseline", "lr_chi2",
    "dt_baseline", "dt_chi2",
    "knn_baseline", "knn_chi2",
]
