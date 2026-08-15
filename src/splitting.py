# splitting.py
# Prepares model inputs by splitting preprocessed data into training and clean test sets.
# Split strategy: stratified by heuristic attack_category (matches notebook Cell 8).
# Output: data/processed/train.csv, data/processed/test_clean.csv

import pandas as pd
from sklearn.model_selection import train_test_split
from . import config


# Classify queries into an expanded set of sub-types and hybrids for stratification.
def assign_attack_category(df):
    df = df.copy()

    def classify(row):
        if row["Label"] == 0:
            return "benign"

        text = str(row["Query"]).lower()

        # Comprehensive heuristic checks for major SQLi vectors
        has_union = "union" in text
        has_bool = "or " in text or "and " in text or "like" in text
        has_error = "extractvalue" in text or "updatexml" in text or "convert(" in text or "cast(" in text
        has_time = "sleep" in text or "benchmark" in text or "waitfor delay" in text or "pg_sleep" in text
        has_stacked = ";" in text and any(cmd in text for cmd in ["select", "update", "insert", "delete", "drop", "alter", "exec"])
        has_oob = "load_file" in text or "into outfile" in text or "into dumpfile" in text

        # Count how many distinct attack characteristics match
        matches = sum([has_union, has_bool, has_error, has_time, has_stacked, has_oob])

        if matches > 1:
            return "hybrid_multitype"
        elif has_union:
            return "union_only"
        elif has_bool:
            return "boolean_only"
        elif has_error:
            return "error_only"
        elif has_time:
            return "time_blind_only"
        elif has_stacked:
            return "stacked_only"
        elif has_oob:
            return "oob_only"
        else:
            return "other_malicious"

    df["attack_category"] = df.apply(classify, axis=1)
    return df


# Load preprocessed corpus from disk.
def load_preprocessed():
    return pd.read_csv(config.PREPROCESSED_PATH)


# Perform multi-type stratified split with safeguards for rare classes.
def run_split():
    # 1. Load preprocessed dataset
    df = load_preprocessed()
    print(f"Loaded {len(df):,} rows from preprocessed dataset")

    # 2. Tag each row with an attack category
    df = assign_attack_category(df)

    # 3. Safeguard: merge any class with < 2 members into 'other_malicious' to allow stratification
    category_counts = df["attack_category"].value_counts()
    rare_categories = category_counts[category_counts < 2].index.tolist()
    if rare_categories:
        print(f"Safeguard: Reassigning rare categories {rare_categories} to 'other_malicious' to allow stratification.")
        df.loc[df["attack_category"].isin(rare_categories), "attack_category"] = "other_malicious"

    print("\nFinal Attack Category Distribution for Stratification:")
    print(df["attack_category"].value_counts())

    # 4. Stratified split by attack category
    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df["attack_category"],
    )

    train_df = train_df.reset_index(drop=True).drop(columns=["attack_category"], errors="ignore")
    test_df = test_df.reset_index(drop=True).drop(columns=["attack_category"], errors="ignore")

    print(f"\nStratified split successful!")
    print(f"Training set: {len(train_df):,} rows")
    print(f"Test set (clean): {len(test_df):,} rows\n")

    # 5. Export split sets to processed directory
    train_df.to_csv(config.TRAIN_PATH, index=False)
    test_df.to_csv(config.TEST_CLEAN_PATH, index=False)

    return train_df, test_df


# Run script independently from command line
if __name__ == "__main__":
    run_split()