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
    chi2_k = select_optimal_k(X_train, y_train, chi2_scores, max_k)
    chi2_indices = top_k_indices(chi2_scores, chi2_k)

    print("Computing Mutual Information scores...")
    mi_scores = compute_mi_scores(X_train, y_train)
    print("Selecting optimal k for Mutual Information...")
    mi_k = select_optimal_k(X_train, y_train, mi_scores, max_k)
    mi_indices = top_k_indices(mi_scores, mi_k)

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