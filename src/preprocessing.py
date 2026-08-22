# preprocessing.py

# Prepares cleaned SQL query text for feature extraction.
#
# Input: data/interim/cleaned.csv
# Output: data/processed/preprocessed.csv

import pandas as pd
from . import config


def load_cleaned():
    # Load the dataset produced by cleaning.py
    return pd.read_csv(config.CLEANED_PATH)


def lowercase_text(df):
    # Normalize all query text to lowercase.
    df = df.copy()
    df["Query"] = df["Query"].astype(str).str.lower()
    return df


def normalize_whitespace(df):
    # Collapse repeated whitespace and remove leading/trailing spaces.
    df = df.copy()
    df["Query"] = (
        df["Query"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df


def drop_empty_after_cleaning(df):
    # Remove queries that are empty after preprocessing.
    before = len(df)
    df = df[df["Query"].str.len() > 0]
    print(f"Dropped {before - len(df):,} rows left empty after preprocessing")
    return df


def preprocess_dataset():
    # Run the preprocessing pipeline and save the result.
    df = load_cleaned()
    print(f"Loaded {len(df):,} rows from cleaned dataset")

    df = lowercase_text(df)
    df = normalize_whitespace(df)
    df = drop_empty_after_cleaning(df)

    df = df.reset_index(drop=True)

    df.to_csv(config.PREPROCESSED_PATH, index=False)

    print(
        f"Saved {len(df):,} preprocessed rows "
        f"(lowercase and whitespace normalized)"
    )

    return df


if __name__ == "__main__":
    preprocess_dataset()