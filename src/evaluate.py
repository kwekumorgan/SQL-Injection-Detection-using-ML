# evaluate_clean.py



import os
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    roc_auc_score,
)

from scipy.stats import chi2 as chi2_dist
from scipy.stats import binomtest

from . import config
from .train import apply_condition


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():
    """
    Load the clean test dataset using the project's
    configured test-data path.
    """

    return pd.read_csv(config.TEST_CLEAN_PATH)


# ============================================================
# LOAD VECTORIZER AND FEATURE CONDITIONS
# ============================================================

def load_artifacts():
    """
    Load the TF-IDF vectorizer and feature-condition information.

    The Chi-square feature indices are shared across all models.
    """

    vectorizer = joblib.load(
        os.path.join(
            config.MODELS_DIR,
            "vectorizer.pkl"
        )
    )

    conditions = joblib.load(
        os.path.join(
            config.MODELS_DIR,
            "feature_conditions.pkl"
        )
    )

    return vectorizer, conditions


# ============================================================
# LOAD ALL TRAINED MODELS
# ============================================================

def load_all_models(model_names, condition_names):
    """
    Load every trained model.

    Each model is stored separately using the naming format:

        model_condition.pkl

    Example:

        nb_baseline.pkl
        nb_chi2.pkl
        svm_baseline.pkl
        svm_chi2.pkl
    """

    models = {}

    for model_name in model_names:

        for condition_name in condition_names:

            key = f"{model_name}_{condition_name}"

            model_path = os.path.join(
                config.MODELS_DIR,
                f"{key}.pkl"
            )

            models[key] = joblib.load(model_path)

    return models


# ============================================================
# GET FEATURE INDICES
# ============================================================

def get_indices_for(conditions, condition_name):
    """
    Return the feature indices for the requested condition.

    The Chi-square feature-index array is shared across
    all classifiers.
    """

    return conditions[condition_name]


# ============================================================
# GET CONTINUOUS SCORES FOR ROC-AUC
# ============================================================

def get_scores_for_auc(model, X):
    """
    Obtain continuous prediction scores for ROC-AUC.

    Models with predict_proba() use the probability of class 1.

    Models such as LinearSVC that do not provide probabilities
    use decision_function().
    """

    if hasattr(model, "predict_proba"):

        return model.predict_proba(X)[:, 1]

    elif hasattr(model, "decision_function"):

        return model.decision_function(X)

    return None


# ============================================================
# COMPUTE METRICS
# ============================================================

def compute_metrics(y_true, y_pred, y_score=None):
    """
    Compute the evaluation metrics for the clean test set.
    """

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f2 = fbeta_score(
        y_true,
        y_pred,
        beta=2,
        zero_division=0
    )

    # False Positive Rate
    if (fp + tn) > 0:

        fpr = fp / (fp + tn)

    else:

        fpr = 0.0

    # Misclassification rate
    total = tp + tn + fp + fn

    if total > 0:

        misclassification_rate = (fp + fn) / total

    else:

        misclassification_rate = 0.0

    # ROC-AUC
    roc_auc = np.nan

    if y_score is not None:

        try:

            roc_auc = roc_auc_score(
                y_true,
                y_score
            )

        except ValueError:

            roc_auc = np.nan

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "f2_score": f2,
        "fpr": fpr,
        "misclassification_rate": misclassification_rate,
        "roc_auc": roc_auc,

        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


# ============================================================
# MCNEMAR'S TEST
# ============================================================

def mcnemar_test(
    y_true,
    pred_baseline,
    pred_chi2
):
    """
    Perform McNemar's paired test comparing:

        baseline vs chi2

    Both predictions must come from the same test rows.

    b:
        baseline correct, chi2 incorrect

    c:
        baseline incorrect, chi2 correct
    """

    y_true = np.asarray(y_true)

    pred_baseline = np.asarray(
        pred_baseline
    )

    pred_chi2 = np.asarray(
        pred_chi2
    )

    baseline_correct = (
        pred_baseline == y_true
    )

    chi2_correct = (
        pred_chi2 == y_true
    )

    # Baseline got the row correct,
    # but Chi-square got it wrong.
    b = int(
        (
            baseline_correct
            & ~chi2_correct
        ).sum()
    )

    # Baseline got the row wrong,
    # but Chi-square got it correct.
    c = int(
        (
            ~baseline_correct
            & chi2_correct
        ).sum()
    )

    discordant = b + c

    # --------------------------------------------------------
    # No discordant pairs
    # --------------------------------------------------------

    if discordant == 0:

        statistic = 0.0
        p_value = 1.0

        method = "No discordant pairs"

    # --------------------------------------------------------
    # Exact binomial McNemar test
    # --------------------------------------------------------

    elif discordant < 25:

        result = binomtest(
            min(b, c),
            n=discordant,
            p=0.5,
            alternative="two-sided"
        )

        statistic = np.nan
        p_value = result.pvalue

        method = "Exact binomial test"

    # --------------------------------------------------------
    # Continuity-corrected McNemar test
    # --------------------------------------------------------

    else:

        statistic = (
            (abs(b - c) - 1) ** 2
        ) / discordant

        p_value = chi2_dist.sf(
            statistic,
            df=1
        )

        method = "Continuity-corrected McNemar"

    alpha = 0.01

    significant = (
        p_value < alpha
    )

    return {
        "b_baseline_better": b,
        "c_chi2_better": c,
        "discordant_pairs": discordant,
        "mcnemar_statistic": statistic,
        "p_value": p_value,
        "alpha": alpha,
        "significant": significant,
        "method": method
    }


# ============================================================
# RUN MCNEMAR TESTS
# ============================================================

def run_mcnemar_tests(
    y_test,
    predictions,
    model_names,
    alpha=0.01
):
    """
    Run McNemar's test for every classifier.

    For each classifier:

        baseline predictions
              vs
        chi2 predictions
    """

    print("\n" + "=" * 90)

    print(
        "MCNEMAR'S TEST: "
        "baseline vs chi2 "
        f"(alpha={alpha})"
    )

    print("=" * 90)

    print(
        f"{'Model':<8}"
        f"{'b':<12}"
        f"{'c':<12}"
        f"{'Discordant':<14}"
        f"{'Statistic':<14}"
        f"{'p-value':<14}"
        f"{'Significant'}"
    )

    print("-" * 90)

    rows = []

    for model_name in model_names:

        y_pred_baseline = predictions[
            (model_name, "baseline")
        ]

        y_pred_chi2 = predictions[
            (model_name, "chi2")
        ]

        result = mcnemar_test(
            y_test,
            y_pred_baseline,
            y_pred_chi2
        )

        statistic = result[
            "mcnemar_statistic"
        ]

        if np.isnan(statistic):

            statistic_text = "exact"

        else:

            statistic_text = f"{statistic:.4f}"

        significant = (
            "YES"
            if result["p_value"] < alpha
            else "no"
        )

        print(
            f"{model_name:<8}"
            f"{result['b_baseline_better']:<12}"
            f"{result['c_chi2_better']:<12}"
            f"{result['discordant_pairs']:<14}"
            f"{statistic_text:<14}"
            f"{result['p_value']:<14.6f}"
            f"{significant}"
        )

        rows.append({
            "model": model_name,
            "b_baseline_better":
                result["b_baseline_better"],
            "c_chi2_better":
                result["c_chi2_better"],
            "discordant_pairs":
                result["discordant_pairs"],
            "mcnemar_statistic":
                result["mcnemar_statistic"],
            "p_value":
                result["p_value"],
            "alpha":
                alpha,
            "significant":
                result["p_value"] < alpha,
            "method":
                result["method"]
        })

    print("-" * 90)

    print(
        "b = rows baseline got right "
        "but chi2 got wrong"
    )

    print(
        "c = rows chi2 got right "
        "but baseline got wrong"
    )

    print(
        "A significant result indicates that "
        "baseline and chi2 differ significantly "
        "in their paired predictions."
    )

    print()

    return pd.DataFrame(rows)


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate_all_models():

    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    test_df = load_test_data()

    # Your dataset uses Query and Label.
    X_text = test_df["Query"]

    y_test = test_df["Label"].values

    # --------------------------------------------------------
    # LOAD VECTORISER AND FEATURE CONDITIONS
    # --------------------------------------------------------

    vectorizer, conditions = load_artifacts()

    # --------------------------------------------------------
    # TRANSFORM TEST QUERIES
    # --------------------------------------------------------

    X_test = vectorizer.transform(
        X_text
    )

    X_test_csc = X_test.tocsc()

    # --------------------------------------------------------
    # FINAL FOUR CLASSIFIERS
    # --------------------------------------------------------

    model_names = [
        "nb",
        "svm",
        "lr",
        "dt"
    ]

    condition_names = [
        "baseline",
        "chi2"
    ]

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    models = load_all_models(
        model_names,
        condition_names
    )

    results = []

    predictions = {}

    # --------------------------------------------------------
    # EVALUATE EACH MODEL
    # --------------------------------------------------------

    for model_name in model_names:

        print("\n" + "=" * 90)

        print(
            f"MODEL: {model_name.upper()}"
        )

        print("=" * 90)

        for condition_name in condition_names:

            key = (
                f"{model_name}_{condition_name}"
            )

            model = models[key]

            # ------------------------------------------------
            # GET FEATURE CONDITION
            # ------------------------------------------------

            indices = get_indices_for(
                conditions,
                condition_name
            )

            X_condition = apply_condition(
                X_test_csc,
                indices
            )

            # ------------------------------------------------
            # MODEL FILE SIZE
            # ------------------------------------------------

            model_path = os.path.join(
                config.MODELS_DIR,
                f"{key}.pkl"
            )

            model_size_kb = (
                os.path.getsize(model_path)
                / 1024
            )

            # ------------------------------------------------
            # PREDICTION + LATENCY
            # ------------------------------------------------

            start_time = time.perf_counter()

            y_pred = model.predict(
                X_condition
            )

            end_time = time.perf_counter()

            total_latency_ms = (
                end_time - start_time
            ) * 1000

            average_latency_ms = (
                total_latency_ms
                / len(y_test)
                if len(y_test) > 0
                else 0.0
            )

            # Store predictions for McNemar's test.
            predictions[
                (model_name, condition_name)
            ] = y_pred

            # ------------------------------------------------
            # ROC-AUC SCORE
            # ------------------------------------------------

            y_score = get_scores_for_auc(
                model,
                X_condition
            )

         
            # COMPUTE METRICS
           

            metrics = compute_metrics(
                y_test,
                y_pred,
                y_score
            )

           
            # SAVE RESULT
           

            result = {
                "model": model_name,

                "condition": condition_name,

                "num_features":
                    X_condition.shape[1],

                "model_size_kb":
                    model_size_kb,

                "total_latency_ms":
                    total_latency_ms,

                "avg_latency_ms":
                    average_latency_ms,

                **metrics
            }

            results.append(result)

            # ------------------------------------------------
            # PRINT RESULT
            # ------------------------------------------------

            print(
                f"\n{key}"
            )

            print(
                f"Accuracy:             "
                f"{metrics['accuracy']:.4f}"
            )

            print(
                f"Precision:            "
                f"{metrics['precision']:.4f}"
            )

            print(
                f"Recall:               "
                f"{metrics['recall']:.4f}"
            )

            print(
                f"F1:                   "
                f"{metrics['f1_score']:.4f}"
            )

            print(
                f"F2:                   "
                f"{metrics['f2_score']:.4f}"
            )

            print(
                f"FPR:                  "
                f"{metrics['fpr']:.4f}"
            )

            print(
                f"Misclassification:    "
                f"{metrics['misclassification_rate']:.4f}"
            )

            if np.isnan(metrics["roc_auc"]):

                print(
                    "ROC-AUC:              N/A"
                )

            else:

                print(
                    f"ROC-AUC:              "
                    f"{metrics['roc_auc']:.4f}"
                )

            print(
                f"TP: {metrics['tp']} | "
                f"TN: {metrics['tn']} | "
                f"FP: {metrics['fp']} | "
                f"FN: {metrics['fn']}"
            )

            print(
                f"Features:             "
                f"{X_condition.shape[1]:,}"
            )

            print(
                f"Model size:           "
                f"{model_size_kb:.2f} KB"
            )

            print(
                f"Total latency:        "
                f"{total_latency_ms:.4f} ms"
            )

            print(
                f"Average latency:      "
                f"{average_latency_ms:.6f} ms"
            )

    # SAVE CLEAN EVALUATION RESULTS
   

    results_df = pd.DataFrame(
        results
    )

    os.makedirs(
        config.METRICS_DIR,
        exist_ok=True
    )

    output_path = os.path.join(
        config.METRICS_DIR,
        "evaluation_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved clean evaluation results to:"
        f"\n{output_path}"
    )

   
    # MCNEMAR'S TEST
    

    mcnemar_df = run_mcnemar_tests(
        y_test,
        predictions,
        model_names,
        alpha=0.01
    )

    mcnemar_output_path = os.path.join(
        config.METRICS_DIR,
        "mcnemar_results.csv"
    )

    mcnemar_df.to_csv(
        mcnemar_output_path,
        index=False
    )

    print(
        f"Saved McNemar results to:"
        f"\n{mcnemar_output_path}"
    )

    return (
        results_df,
        mcnemar_df
    )


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":

    evaluate_all_models()