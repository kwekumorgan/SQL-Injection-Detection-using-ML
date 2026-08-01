# feature_selection.py
# Computes Chi-square and Mutual Information feature scores and optimizes
# the feature count (top-k) using cross-validated baseline classification.

import numpy as np
from sklearn.feature_selection import chi2, mutual_info_classif
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


# Evaluate mean CV accuracy for top-k features on CSC-formatted matrix.
def evaluate_k(X_train_csc, y_train, scores, k):
    # Slice columns using CSC matrix for fast indexing across iterations
    indices = top_k_indices(scores, k)
    X_subset = X_train_csc[:, indices]

    model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)
    result = cross_val_score(model, X_subset, y_train, cv=5, scoring="accuracy")
    return result.mean()