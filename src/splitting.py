# splitting.py
# Splits preprocessed data into train and clean test sets.
# Stratified 80/20 split, no separate validation set (CV handles that now).
# Output: data/processed/train.csv, data/processed/test_clean.csv

import re
import pandas as pd
from sklearn.model_selection import train_test_split
from . import config


# Classify queries into attack sub-types for stratification.
def assign_attack_category(df):
    df = df.copy()

    # Tag one row based on which attack patterns it matches.
    def classify(row):
        if row["Label"] == 0:
            return "benign"

        text = str(row["Query"]).lower()
        matches = []

        for attack_type, pattern in config.ATTACK_TYPE_PATTERNS.items():
            if re.search(pattern, text):
                matches.append(attack_type)

        # Multiple matches means the query combines attack types.
        if len(matches) > 1:
            return "hybrid_multitype"
        elif len(matches) == 1:
            return f"{matches[0].lower()}_only"
        else:
            return "other_malicious"

    df["attack_category"] = df.apply(classify, axis=1)
    return df


# Load preprocessed corpus from disk.
def load_preprocessed():
    return pd.read_csv(config.PREPROCESSED_PATH)


# Stratified 80/20 split with safeguard for rare classes.
def run_split():
    df = load_preprocessed()
    print(f"Loaded {len(df):,} rows from preprocessed dataset")

    df = assign_attack_category(df)

    # Merge rare classes into 'other_malicious' to allow stratification.
    category_counts = df["attack_category"].value_counts()
    rare_categories = category_counts[category_counts < 10].index.tolist()

    if rare_categories:
        print(f"Reassigning rare categories {rare_categories} to 'other_malicious'.")
        df.loc[df["attack_category"].isin(rare_categories), "attack_category"] = "other_malicious"

    print("\nAttack category distribution:")
    print(df["attack_category"].value_counts())

    # config.TEST_SIZE should be 0.2 for a true 80/20 split.
    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df["attack_category"],
    )

    # Drop the helper column, it was only needed for stratification.
    train_df = train_df.reset_index(drop=True).drop(columns=["attack_category"], errors="ignore")
    test_df = test_df.reset_index(drop=True).drop(columns=["attack_category"], errors="ignore")

    print("\nSplit complete!")
    print(f"Training set: {len(train_df):,} rows")
    print(f"Test set:     {len(test_df):,} rows\n")

    train_df.to_csv(config.TRAIN_PATH, index=False)
    test_df.to_csv(config.TEST_CLEAN_PATH, index=False)

    return train_df, test_df


# Run script directly from the command line.
if __name__ == "__main__":
    run_split()