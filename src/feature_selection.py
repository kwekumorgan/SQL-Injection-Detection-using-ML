# feature_selection.py
# Implements Chi-squared feature scoring and CV-based grid search for optimal top-k features.

# imports
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import cross_val_score

# Custom F2 scorer prioritizing Recall for malicious query detection
F2_SCORER = make_scorer(fbeta_score, beta=2, zero_division=0)


# Compute Chi-squared statistics for each feature
def compute_chi2_scores(X_train, y_train):
    scores, _ = chi2(X_train, y_train)
    return scores


# Return feature indices sorted by highest score first
def top_k_indices(scores, k):
    return np.argsort(scores)[::-1][:k]


# Score a specific k value using k-fold cross-validation
def evaluate_k_cv(X_train_csr, y_train, scores, k, model, cv):
    indices = top_k_indices(scores, k)
    X_subset = X_train_csr[:, indices]
    cv_scores = cross_val_score(model, X_subset, y_train, cv=cv, scoring=F2_SCORER)
    return cv_scores.mean()


# Coarse grid search over k range in larger increments
def coarse_search_cv_ranged(X_train_csr, y_train, scores, k_min, k_max, model, cv, step=100):
    step = min(step, max(1, k_max - k_min))
    results = {}
    for k in range(k_min, k_max + 1, step):
        results[k] = evaluate_k_cv(X_train_csr, y_train, scores, k, model, cv)
        print(f"   k={k:,}  cv_f2={results[k]:.4f}")
    best_k = max(results, key=results.get)
    return best_k, results


# Fine grid search around coarse best k in smaller increments
def fine_search_cv_ranged(X_train_csr, y_train, scores, best_coarse_k, k_min, k_max, model, cv, window=200, step=10):
    lower = max(k_min, best_coarse_k - window)
    upper = min(k_max, best_coarse_k + window)
    results = {}
    for k in range(lower, upper + 1, step):
        results[k] = evaluate_k_cv(X_train_csr, y_train, scores, k, model, cv)
        print(f"   k={k:,}  cv_f2={results[k]:.4f}")
    best_k = max(results, key=results.get)
    return best_k, results


# Identify contiguous score plateau around peak and return midpoint k
def find_plateau_around_best(results, tolerance=0.005):
    best_k = max(results, key=results.get)
    best_score = results[best_k]
    threshold = best_score - tolerance

    sorted_ks = sorted(results.keys())
    idx = sorted_ks.index(best_k)

    left = idx
    while left > 0 and results[sorted_ks[left - 1]] >= threshold:
        left -= 1

    right = idx
    while right < len(sorted_ks) - 1 and results[sorted_ks[right + 1]] >= threshold:
        right += 1

    plateau_start = sorted_ks[left]
    plateau_end = sorted_ks[right]
    midpoint = (plateau_start + plateau_end) // 2
    return midpoint, plateau_start, plateau_end


# Search optimal k using coarse/fine CV search and plateau midpoint selection
def select_optimal_k_cv_ranged(X_train, y_train, scores, k_min, k_max, model, cv, tolerance=0.005):
    X_train_csr = X_train.tocsr()
    k_max = min(k_max, X_train_csr.shape[1])
    print(f"Searching k in [{k_min:,}, {k_max:,}]\n")

    coarse_best, _ = coarse_search_cv_ranged(X_train_csr, y_train, scores, k_min, k_max, model, cv, step=100)
    fine_best, fine_results = fine_search_cv_ranged(X_train_csr, y_train, scores, coarse_best, k_min, k_max, model, cv, window=200, step=10)
    mid_k, plateau_start, plateau_end = find_plateau_around_best(fine_results, tolerance=tolerance)

    print(f"\nPlateau: [{plateau_start:,}, {plateau_end:,}] | Selected k: {mid_k:,}")
    return mid_k, fine_results


# Determine shared k across models using Naive Bayes as baseline
def select_shared_k_via_nb(X_train, y_train, scores, max_k, cv, tolerance=0.005):
    nb_model = MultinomialNB()
    return select_optimal_k_cv_ranged(X_train, y_train, scores, k_min=1000, k_max=max_k, model=nb_model, cv=cv, tolerance=tolerance)