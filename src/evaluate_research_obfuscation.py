# evaluate_research_obfuscation.py
# Evaluate clean and obfuscated SQLi detection under baseline and Chi-square conditions.

import os
import numpy as np
import joblib
import pandas as pd
from scipy.stats import chi2 as chi2_dist, binomtest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from . import config


MODEL_NAMES = ["nb", "svm", "lr", "dt"]


# Load trained artifacts
def load_artifacts():
    vectorizer = joblib.load(
        os.path.join(config.MODELS_DIR, "vectorizer.pkl")
    )
    conditions = joblib.load(
        os.path.join(config.MODELS_DIR, "feature_conditions.pkl")
    )
    return vectorizer, conditions


def get_indices_for(conditions, condition_name):
    return conditions[condition_name]


# Transform queries using the fitted feature representation
def transform_features(vectorizer, selected_indices, df, text_col="Query"):
    X_full = vectorizer.transform(df[text_col].astype(str))

    if selected_indices is None:
        return X_full

    return X_full[:, selected_indices]


# Load trained model
def load_model(model_name, condition_name):
    key = f"{model_name}_{condition_name}"
    path = os.path.join(config.MODELS_DIR, f"{key}.pkl")

    if not os.path.exists(path):
        print(f"  Model not found, skipping: {path}")
        return None, key

    return joblib.load(path), key


# Calculate evaluation metrics
def compute_full_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
    }


# McNemar paired test
def mcnemar_test(y_true, y_pred_a, y_pred_b):
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    n = b + c

    if n == 0:
        return b, c, None, 1.0

    # Exact test for small discordant counts
    if n < 25:
        result = binomtest(
            min(b, c),
            n,
            0.5,
            alternative="two-sided"
        )
        return b, c, None, result.pvalue

    # Continuity-corrected McNemar statistic
    statistic = (abs(b - c) - 1) ** 2 / n
    p_value = 1 - chi2_dist.cdf(
        statistic,
        df=1
    )

    return b, c, statistic, p_value


# Compare clean and obfuscated predictions within each condition
def run_per_condition_degradation_tests(
    y_true_clean,
    y_true_obf,
    clean_predictions,
    obf_predictions,
    model_names,
    alpha=0.01
):
    print("\n" + "=" * 100)
    print(
        f"MCNEMAR'S TEST: clean vs obfuscated, "
        f"WITHIN each condition -- alpha={alpha}"
    )
    print("=" * 100)

    print(
        f"{'Model':<8}"
        f"{'Condition':<12}"
        f"{'b (clean>obf)':<16}"
        f"{'c (obf>clean)':<16}"
        f"{'statistic':<12}"
        f"{'p-value':<12}"
        f"{'significant'}"
    )
    print("-" * 100)

    rows = []

    for model_name in model_names:
        for condition_name in ["baseline", "chi2"]:
            key = (model_name, condition_name)

            if key not in clean_predictions or key not in obf_predictions:
                continue

            y_pred_clean = clean_predictions[key]
            y_pred_obf = obf_predictions[key]

            b, c, statistic, p_value = mcnemar_test(
                y_true_clean,
                y_pred_clean,
                y_pred_obf
            )

            significant = "YES" if p_value < alpha else "no"
            stat_str = (
                f"{statistic:.4f}"
                if statistic is not None
                else "exact"
            )

            print(
                f"{model_name:<8}"
                f"{condition_name:<12}"
                f"{b:<16}"
                f"{c:<16}"
                f"{stat_str:<12}"
                f"{p_value:<12.4f}"
                f"{significant}"
            )

            rows.append({
                "model": model_name,
                "condition": condition_name,
                "b_correct_clean_wrong_obf": b,
                "c_wrong_clean_correct_obf": c,
                "statistic": statistic,
                "p_value": p_value,
                f"significant_at_{alpha}": p_value < alpha,
            })

    print("-" * 100)
    print("b = correct on clean, wrong on obfuscated")
    print("c = wrong on clean, correct on obfuscated\n")

    return pd.DataFrame(rows)


# Main evaluation
def evaluate_research_obfuscation():
    # Load artifacts and test sets
    vectorizer, conditions = load_artifacts()

    clean_df = pd.read_csv(
        config.TEST_CLEAN_PATH
    )

    obf_df = pd.read_csv(
        config.RESEARCH_OBFUSCATED_PATH
    )

    print(
        f"Clean rows: {len(clean_df):,}  "
        f"Obfuscated rows: {len(obf_df):,}"
    )

    # Validate paired test sets
    if len(clean_df) != len(obf_df):
        raise ValueError(
            "Clean and obfuscated test sets must have the same number of rows "
            "for paired McNemar tests. Got "
            f"{len(clean_df)} vs {len(obf_df)}."
        )

    results = []
    clean_predictions = {}
    obf_predictions = {}

    # Evaluate all classifiers under both feature conditions
    for model_name in MODEL_NAMES:
        for condition_name in conditions.keys():

            model, model_key = load_model(
                model_name,
                condition_name
            )

            if model is None:
                continue

            selected_indices = get_indices_for(
                conditions,
                condition_name
            )

            # Transform clean and obfuscated test data
            X_clean = transform_features(
                vectorizer,
                selected_indices,
                clean_df
            )

            X_obf = transform_features(
                vectorizer,
                selected_indices,
                obf_df
            )

            # Generate predictions
            y_pred_clean = model.predict(X_clean)
            y_pred_obf = model.predict(X_obf)

            clean_predictions[
                (model_name, condition_name)
            ] = y_pred_clean

            obf_predictions[
                (model_name, condition_name)
            ] = y_pred_obf

            # Calculate metrics
            clean_metrics = compute_full_metrics(
                clean_df["Label"],
                y_pred_clean
            )

            obf_metrics = compute_full_metrics(
                obf_df["Label"],
                y_pred_obf
            )

            print(f"{model_key}:")

            print(
                f"  clean: "
                f"acc={clean_metrics['accuracy']:.4f}  "
                f"prec={clean_metrics['precision']:.4f}  "
                f"rec={clean_metrics['recall']:.4f}  "
                f"f1={clean_metrics['f1']:.4f}  "
                f"fpr={clean_metrics['fpr']:.4f}"
            )

            print(
                f"  obf:   "
                f"acc={obf_metrics['accuracy']:.4f}  "
                f"prec={obf_metrics['precision']:.4f}  "
                f"rec={obf_metrics['recall']:.4f}  "
                f"f1={obf_metrics['f1']:.4f}  "
                f"fpr={obf_metrics['fpr']:.4f}"
            )

            # Store metrics and clean-to-obfuscated changes
            results.append({
                "model": model_name,
                "condition": condition_name,

                "clean_accuracy": clean_metrics["accuracy"],
                "obf_accuracy": obf_metrics["accuracy"],

                "clean_precision": clean_metrics["precision"],
                "obf_precision": obf_metrics["precision"],

                "clean_recall": clean_metrics["recall"],
                "obf_recall": obf_metrics["recall"],

                "clean_f1": clean_metrics["f1"],
                "obf_f1": obf_metrics["f1"],

                "clean_fpr": clean_metrics["fpr"],
                "obf_fpr": obf_metrics["fpr"],

                "recall_delta": (
                    obf_metrics["recall"]
                    - clean_metrics["recall"]
                ),

                "precision_delta": (
                    obf_metrics["precision"]
                    - clean_metrics["precision"]
                ),

                "fpr_delta": (
                    obf_metrics["fpr"]
                    - clean_metrics["fpr"]
                ),
            })

    # Save performance results
    results_df = pd.DataFrame(results)

    os.makedirs(
        config.METRICS_DIR,
        exist_ok=True
    )

    output_path = os.path.join(
        config.METRICS_DIR,
        "research_obfuscation_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(f"\nSaved -> {output_path}")

    # Run clean-versus-obfuscated statistical tests
    mcnemar_df = run_per_condition_degradation_tests(
        clean_df["Label"].values,
        obf_df["Label"].values,
        clean_predictions,
        obf_predictions,
        MODEL_NAMES,
    )

    # Save statistical results
    mcnemar_output_path = os.path.join(
        config.METRICS_DIR,
        "research_obfuscation_mcnemar.csv"
    )

    mcnemar_df.to_csv(
        mcnemar_output_path,
        index=False
    )

    print(f"Saved -> {mcnemar_output_path}")

    return results_df, mcnemar_df


if __name__ == "__main__":
    evaluate_research_obfuscation()