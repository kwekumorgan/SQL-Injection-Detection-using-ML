# train.py
# Trains baseline classifiers across three feature selection conditions.

import os
import joblib
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from . import config
from .features import build_vectorizer, fit_transform_train
from .feature_selection import (
    compute_chi2_scores,
    compute_mi_scores,
    select_optimal_k,
    top_k_indices,
    smallest_k_within_tolerance,   # --- ADDED ---
)


# Instantiate base classification models for experiment.
def get_models():
    return {
        "nb": MultinomialNB(),
        "svm": LinearSVC(random_state=config.RANDOM_STATE, max_iter=5000, dual="auto"),
        "lr": LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
        "dt": DecisionTreeClassifier(random_state=config.RANDOM_STATE),
        "knn": KNeighborsClassifier(),
    }


# Compute Chi-square and MI scores to select optimal feature subsets.
def build_feature_conditions(X_train, y_train):
    max_k = X_train.shape[1]

    print("Computing Chi-square scores...")
    chi2_scores = compute_chi2_scores(X_train, y_train)
    print("Selecting optimal k for Chi-square...")
    chi2_k, chi2_coarse, chi2_fine = select_optimal_k(X_train, y_train, chi2_scores, max_k)
    # --- ADDED: use the smallest k within tolerance of the peak, instead of the raw peak k ---
    efficient_chi2_k = smallest_k_within_tolerance(chi2_fine, tolerance=0.005)
    print(f"Chi-square: raw best k={chi2_k:,}  ->  efficient k={efficient_chi2_k:,}")
    chi2_indices = top_k_indices(chi2_scores, efficient_chi2_k)

    print("Computing Mutual Information scores...")
    mi_scores = compute_mi_scores(X_train, y_train)
    print("Selecting optimal k for Mutual Information...")
    mi_k, mi_coarse, mi_fine = select_optimal_k(X_train, y_train, mi_scores, max_k)
    # --- ADDED: same tolerance logic for MI ---
    efficient_mi_k = smallest_k_within_tolerance(mi_fine, tolerance=0.005)
    print(f"Mutual Information: raw best k={mi_k:,}  ->  efficient k={efficient_mi_k:,}")
    mi_indices = top_k_indices(mi_scores, efficient_mi_k)

    return {
        "baseline": None,
        "chi2": chi2_indices,
        "mi": mi_indices,
    }


# Slice feature matrix to designated indices or retain full feature space.
def apply_condition(X_csc, indices):
    if indices is None:
        return X_csc
    return X_csc[:, indices]



# Converts feature matrix to CSC format once for fast column slicing,
# then trains all 5 classifiers across each of the 3 feature conditions (15 models total).
def train_all_models(X_train, y_train, conditions):
    trained = {}
    X_train_csc = X_train.tocsc()  # Optimized sparse format for column-wise operations

    for condition_name, indices in conditions.items():
        X_condition = apply_condition(X_train_csc, indices)
        print(f"\nTraining condition: {condition_name} ({X_condition.shape[1]:,} features)")

        # Iterate over fresh model instances to avoid reusing fitted models
        for model_name, model in get_models().items():
            print(f"   Fitting {model_name}...")
            model.fit(X_condition, y_train)
            trained[f"{model_name}_{condition_name}"] = model

    return trained



# Persists all 15 trained model files, the fitted vectorizer object,
# and selected feature indices mapping into the target models directory.
def save_models(trained_models, vectorizer, conditions):
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # Export fitted classifiers individually
    for name, model in trained_models.items():
        path = os.path.join(config.MODELS_DIR, f"{name}.pkl")
        joblib.dump(model, path)

    # Export vectorizer and feature subset metadata needed for evaluation/testing
    joblib.dump(vectorizer, os.path.join(config.MODELS_DIR, "vectorizer.pkl"))
    joblib.dump(conditions, os.path.join(config.MODELS_DIR, "feature_conditions.pkl"))

    print(f"\nSaved {len(trained_models)} models, vectorizer, and feature conditions to {config.MODELS_DIR}")


# Pipeline entrypoint: loads training data, extracts features, calculates selection
# conditions, fits all classifiers, and saves generated artifacts.
def run_training():
    train_df = pd.read_csv(config.TRAIN_PATH)
    y_train = train_df["Label"]

    vectorizer = build_vectorizer()
    X_train = fit_transform_train(vectorizer, train_df)

    conditions = build_feature_conditions(X_train, y_train)
    trained_models = train_all_models(X_train, y_train, conditions)
    save_models(trained_models, vectorizer, conditions)

    return trained_models, vectorizer, conditions


# Allow script execution directly from terminal/command line
if __name__ == "__main__":
    run_training()