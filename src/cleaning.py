# cleaning.py

import pandas as pd
from . import config


def load_merged():
  # load the dataset produced by data_acquisition.py
  return pd.read_csv(config.MERGED_PATH)


def handle_missing_values(df):
    # drop rows with missing query or label, empty strings, or unparseable Excel artifacts
    before = len(df)
    
    # 1. Drop NaN / null entries
    df = df.dropna(subset=["Query", "Label"])
    
    # 2. Filter out corrupted Excel formula artifacts and blank strings
    excel_artifacts = ["#NAME?", "#VALUE!", "#REF!", "#N/A", "nan", "null"]
    df["Query"] = df["Query"].astype(str).str.strip()
    df = df[~df["Query"].str.upper().isin(excel_artifacts)]
    df = df[df["Query"].str.len() > 0]
    
    print(f"Dropped {before - len(df):,} rows with missing values or invalid artifacts")
    return df


def fix_label_encoding(df):
    # Convert labels to numeric values; invalid values become NaN
    df = df.copy()
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")

    # Remove rows where the label could not be converted
    before = len(df)
    df = df.dropna(subset=["Label"])
    print(f"Dropped {before - len(df):,} rows with invalid label values")

    # Convert valid numeric labels to integers
    df["Label"] = df["Label"].astype(int)

    # Keep only the two classes used in the binary SQLi experiment
    before = len(df)
    df = df[df["Label"].isin([0, 1])]
    print(f"Dropped {before - len(df):,} rows with non-binary labels")

    return df


def standardize_formatting(df):
  # ensure query text is string type
  df = df.copy()
  df["Query"] = df["Query"].astype(str)

  # remove leading and trailing whitespace
  df["Query"] = df["Query"].str.strip()

  # collapse repeated inner whitespace into a single space
  df["Query"] = df["Query"].str.replace(r"\s+", " ", regex=True)

  # drop rows left empty after cleaning
  before = len(df)
  df = df[df["Query"].str.len() > 0]
  print(f"Dropped {before - len(df):,} rows with empty text")
  return df


def remove_conflicting_labels(df):
  # count how many distinct labels each query appears under
  label_counts = df.groupby("Query")["Label"].nunique()

  # identify queries that appear under more than one label
  conflicting_queries = label_counts[label_counts > 1].index

  # drop all rows for those queries, since the correct label is ambiguous
  before = len(df)
  df = df[~df["Query"].isin(conflicting_queries)]
  print(f"Dropped {before - len(df):,} rows with conflicting labels")
  return df


def remove_duplicates(df):
  # remove exact duplicate rows, keeping the first occurrence
  before = len(df)
  df = df.drop_duplicates(subset=["Query"])
  print(f"Dropped {before - len(df):,} exact duplicate rows")
  return df


def clean_dataset():
  # run the full cleaning pipeline in order
  df = load_merged()
  print(f"Loaded {len(df):,} rows from merged dataset")

  df = handle_missing_values(df)
  df = fix_label_encoding(df)
  df = standardize_formatting(df)
  df = remove_conflicting_labels(df)
  df = remove_duplicates(df)

  # reset index after dropping rows so it stays continuous
  df = df.reset_index(drop=True)
  print(f"Final cleaned dataset: {len(df):,} rows")

  df.to_csv(config.CLEANED_PATH, index=False)
  return df


if __name__ == "__main__":
  clean_dataset()