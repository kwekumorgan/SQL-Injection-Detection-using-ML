# feature_selection.py
# Computes Chi-square and Mutual Information feature scores and optimizes
# the feature count (top-k) using cross-validated baseline classification.

import numpy as np
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from . import config


# Compute Chi-squared statistic scores for input features.
def compute_chi2_scores(X_train, y_train):
    scores, _ = chi2(X_train, y_train)
    return scores


# Compute Mutual Information scores for input features.
def compute_mi_scores(X_train, y_train):
    return mutual_info_classif(X_train, y_train, random_state=config.RANDOM_STATE)


# Return feature indices sorted by highest score in descending order.
def top_k_indices(scores, k):
    return np.argsort(scores)[::-1][:k]


# Evaluate mean CV F1 for top-k features on CSC-formatted matrix.
def evaluate_k(X_train_csc, y_train, scores, k, model=None):
    # Slice columns using CSC matrix for fast indexing across iterations
    indices = top_k_indices(scores, k)
    X_subset = X_train_csc[:, indices]

    # Fallback to MultinomialNB if no model is provided
    if model is None:
        model = MultinomialNB()

    result = cross_val_score(model, X_subset, y_train, cv=5, scoring="f1")
    return result.mean()


# Perform coarse grid search across candidate k values at step intervals.
def coarse_search(X_train_csc, y_train, scores, max_k, step=50, model=None):
    step = min(step, max_k)  # Guard against vocabulary smaller than step size
    results = {}

    for k in range(step, max_k + 1, step):
        results[k] = evaluate_k(X_train_csc, y_train, scores, k, model=model)
        print(f"   k={k:,}  f1={results[k]:.4f}")


    best_k = max(results, key=results.get)
    return best_k, results


# Perform fine grid search around the best coarse k value (step size 1, +/-200 window).
def fine_search(X_train_csc, y_train, scores, best_coarse_k, window=200, model=None):
    max_k = X_train_csc.shape[1]
    lower = max(1, best_coarse_k - window)
    upper = min(max_k, best_coarse_k + window)
    results = {}


    for k in range(lower, upper + 1):
        results[k] = evaluate_k(X_train_csc, y_train, scores, k, model=model)
        if k % 20 == 0:  # Print progress every 20 steps
            print(f"   k={k:,}  f1={results[k]:.4f}")

    best_k = max(results, key=results.get)
    return best_k, results


# Execute two-stage search pipeline to find optimal feature count k.
def select_optimal_k(X_train, y_train, scores, max_k, model=None):
    # Enforce sparse input matrix requirement
    if not hasattr(X_train, "tocsc"):
        raise TypeError(
            f"X_train must be a scipy sparse matrix (e.g. from TfidfVectorizer), "
            f"got {type(X_train).__name__} instead."
        )

    # Convert to CSC once for efficient column slicing
    X_train_csc = X_train.tocsc()

    print("Running coarse search (step size = 50)...")
    coarse_best, coarse_results = coarse_search(
        X_train_csc, y_train, scores, max_k, step=50, model=model
    )
    print(f"Coarse search best k: {coarse_best:,}\n")

    print("Running fine search (+/-200 window, step size = 1)...")
    fine_best, fine_results = fine_search(
        X_train_csc, y_train, scores, coarse_best, window=200, model=model
    )
    print(f"Fine search best k: {fine_best:,}")

    return fine_best, coarse_results, fine_results