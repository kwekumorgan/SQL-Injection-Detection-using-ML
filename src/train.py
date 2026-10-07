# train.py
# Script to train baseline and Chi-squared feature-selected classification models.

# imports
import os
import warnings
import joblib
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

from . import config
from .features import build_vectorizer, fit_transform_train
from .feature_selection import (
    compute_chi2_scores,
    select_shared_k_via_nb,
    top_k_indices,
)

warnings.filterwarnings("ignore", category=UserWarning)


# Instantiate base classification models
def get_models():
    return {
        "nb": MultinomialNB(),
        "svm": LinearSVC(random_state=config.RANDOM_STATE, max_iter=5000, dual="auto"),
        "lr": LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
        "dt": DecisionTreeClassifier(random_state=config.RANDOM_STATE),
        "knn": KNeighborsClassifier(),
    }


# Build feature selection conditions (full baseline vs chi2 optimal k)
def build_feature_conditions(X_train, y_train):
    print("\nComputing Chi-square scores...")
    chi2_scores = compute_chi2_scores(X_train, y_train)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
    search_k_max = min(X_train.shape[1], 10000)

    print(f"\nSelecting shared k via NB (CV, range [1000, {search_k_max:,}])...")
    chi2_k, _ = select_shared_k_via_nb(X_train, y_train, chi2_scores, search_k_max, cv)
    print(f"Selected shared k: {chi2_k:,}")

    chi2_indices = top_k_indices(chi2_scores, chi2_k)

    return {
        "baseline": None,
        "chi2": chi2_indices,
    }


# Slice feature matrix to selected indices or keep full features
def apply_condition(X_csc, indices):
    if indices is None:
        return X_csc
    return X_csc[:, indices]


# Train baseline and chi2 variants for each classifier
def train_all_models(X_train, y_train, conditions):
    trained = {}
    X_train_csc = X_train.tocsc()

    for condition_name, indices in conditions.items():
        print(f"\nTraining condition: {condition_name}")

        for model_name, model in get_models().items():
            X_condition = apply_condition(X_train_csc, indices)

            print(f"   Fitting {model_name} ({X_condition.shape[1]:,} features)...")
            model.fit(X_condition, y_train)

            trained[f"{model_name}_{condition_name}"] = model

    return trained


# Save trained models, vectorizer, and feature conditions to disk
def save_models(trained_models, vectorizer, conditions):
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    for name, model in trained_models.items():
        joblib.dump(model, os.path.join(config.MODELS_DIR, f"{name}.pkl"))

    joblib.dump(vectorizer, os.path.join(config.MODELS_DIR, "vectorizer.pkl"))
    joblib.dump(conditions, os.path.join(config.MODELS_DIR, "feature_conditions.pkl"))

    print(f"\nSaved {len(trained_models)} models, vectorizer, and feature conditions to {config.MODELS_DIR}")


# Pipeline execution: load training set, select features, fit, and save artifacts
def run_training():
    train_df = pd.read_csv(config.RANDOMIZED_TRAIN_PATH)
    y_train = train_df["Label"]

    vectorizer = build_vectorizer()
    X_train = fit_transform_train(vectorizer, train_df)

    conditions = build_feature_conditions(X_train, y_train)
    trained_models = train_all_models(X_train, y_train, conditions)
    save_models(trained_models, vectorizer, conditions)

    return trained_models, vectorizer, conditions


if __name__ == "__main__":
    run_training()