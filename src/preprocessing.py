# preprocessing.py
# Normalizes cleaned SQL query text for feature extraction.

# imports
import pandas as pd
from . import config


# Load intermediate cleaned dataset
def load_cleaned():
    return pd.read_csv(config.CLEANED_PATH)


# Convert query text to lowercase
def lowercase_text(df):
    df = df.copy()
    df["Query"] = df["Query"].astype(str).str.lower()
    return df


# Collapse multi-spaces and strip leading/trailing whitespace
def normalize_whitespace(df):
    df = df.copy()
    df["Query"] = (
        df["Query"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df


# Drop records reduced to empty strings after normalization
def drop_empty_after_cleaning(df):
    before = len(df)
    df = df[df["Query"].str.len() > 0]
    print(f"Dropped {before - len(df):,} rows left empty after preprocessing")
    return df


# Log class balance breakdown for benign and malicious labels
def log_label_distribution(df, name):
    counts = df["Label"].value_counts().sort_index()
    total = len(df)
    print(f"  {name} label distribution:")
    for label, count in counts.items():
        pct = (count / total) * 100
        tag = "benign" if label == 0 else "malicious"
        print(f"    Label {label} ({tag}): {count:,} ({pct:.1f}%)")


# Execute text preprocessing pipeline and export result
def preprocess_dataset():
    df = load_cleaned()
    print(f"Loaded {len(df):,} rows from cleaned dataset")

    log_label_distribution(df, "Loaded (pre-preprocessing)")

    # Execute text cleaning transformations
    df = lowercase_text(df)
    df = normalize_whitespace(df)
    df = drop_empty_after_cleaning(df)

    df = df.reset_index(drop=True)
    log_label_distribution(df, "Final (post-preprocessing)")

    # Export preprocessed dataset
    df.to_csv(config.PREPROCESSED_PATH, index=False)
    print(f"Saved {len(df):,} preprocessed rows")

    return df


if __name__ == "__main__":
    preprocess_dataset()