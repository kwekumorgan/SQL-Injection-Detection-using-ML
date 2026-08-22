# feature_selection.py
# Computes Chi-square and Mutual Information feature scores and optimizes
# the feature count (top-k) using a validation split.

import numpy as np
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from . import config


# Compute Chi-squared statistic scores for input features.
def compute_chi2_scores(X_train, y_train):
    scores, _ = chi2(X_train, y_train)
    return scores


# Compute Mutual Information scores for input features.
def compute_mi_scores(X_train, y_train):
    return mutual_info_classif(
        X_train,
        y_train,
        random_state=config.RANDOM_STATE
    )


# Return feature indices sorted by highest score in descending order.
def top_k_indices(scores, k):
    return np.argsort(scores)[::-1][:k]


# Evaluate F1 for top-k features using a separate validation set.
def evaluate_k(X_train_csc, y_train, X_valid_csc, y_valid, scores, k, model=None):
    # Slice columns using CSC matrix for fast indexing across iterations
    indices = top_k_indices(scores, k)

    X_train_subset = X_train_csc[:, indices]
    X_valid_subset = X_valid_csc[:, indices]

    # Fallback to MultinomialNB if no model is provided
    if model is None:
        model = MultinomialNB()

    model.fit(X_train_subset, y_train)

    predictions = model.predict(X_valid_subset)

    return f1_score(y_valid, predictions)


# Perform coarse grid search across candidate k values at step intervals.
def coarse_search(
    X_train_csc,
    y_train,
    X_valid_csc,
    y_valid,
    scores,
    max_k,
    step=50,
    model=None
):
    step = min(step, max_k)  # Guard against vocabulary smaller than step size
    results = {}

    for k in range(step, max_k + 1, step):
        results[k] = evaluate_k(
            X_train_csc,
            y_train,
            X_valid_csc,
            y_valid,
            scores,
            k,
            model=model
        )
        print(f"   k={k:,}  f1={results[k]:.4f}")

    best_k = max(results, key=results.get)
    return best_k, results


# Perform fine grid search around the best coarse k value (step size 1, +/-200 window).
def fine_search(
    X_train_csc,
    y_train,
    X_valid_csc,
    y_valid,
    scores,
    best_coarse_k,
    window=200,
    model=None
):
    max_k = X_train_csc.shape[1]
    lower = max(1, best_coarse_k - window)
    upper = min(max_k, best_coarse_k + window)
    results = {}

    for k in range(lower, upper + 1):
        results[k] = evaluate_k(
            X_train_csc,
            y_train,
            X_valid_csc,
            y_valid,
            scores,
            k,
            model=model
        )

        if k % 20 == 0:  # Print progress every 20 steps
            print(f"   k={k:,}  f1={results[k]:.4f}")

    best_k = max(results, key=results.get)
    return best_k, results


# Find the smallest k whose F1 score is within `tolerance` of the best score.
def smallest_k_within_tolerance(results, tolerance=0.005):
    best_score = max(results.values())

    candidates = [
        k for k, score in results.items()
        if score >= best_score - tolerance
    ]

    return min(candidates)


# Execute two-stage search pipeline to find optimal feature count k.
def select_optimal_k(X_train, y_train, scores, max_k, model=None):
    # Enforce sparse input matrix requirement
    if not hasattr(X_train, "tocsc"):
        raise TypeError(
            f"X_train must be a scipy sparse matrix (e.g. from TfidfVectorizer), "
            f"got {type(X_train).__name__} instead."
        )

    # Split training data into fitting and validation portions
    # for selecting the optimal feature count.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_train
    )

    # Convert to CSC for efficient column slicing
    X_train_csc = X_train.tocsc()
    X_valid_csc = X_valid.tocsc()

    # Cap max_k at 2500 or the total vocabulary size, whichever is smaller
    max_k = min(max_k, 2500, X_train_csc.shape[1])
    print(f"Capped maximum search space (max_k) to: {max_k:,}\n")

    if model is None:
        model = MultinomialNB()

    print("Running coarse search (step size = 50)...")
    coarse_best, coarse_results = coarse_search(
        X_train_csc,
        y_train,
        X_valid_csc,
        y_valid,
        scores,
        max_k,
        step=50,
        model=model
    )
    print(f"Coarse search best k: {coarse_best:,}\n")

    print("Running fine search (+/-200 window, step size = 1)...")
    fine_best, fine_results = fine_search(
        X_train_csc,
        y_train,
        X_valid_csc,
        y_valid,
        scores,
        coarse_best,
        window=200,
        model=model
    )
    print(f"Fine search best k: {fine_best:,}")

    # Select the smallest feature count whose F1 score is close
    # enough to the best observed result.
    efficient_k = smallest_k_within_tolerance(
        fine_results,
        tolerance=0.005
    )

    print(
        f"Efficient k selected: {efficient_k:,} "
        f"(within 0.005 F1 of the best result)"
    )

    return efficient_k, coarse_results, fine_results