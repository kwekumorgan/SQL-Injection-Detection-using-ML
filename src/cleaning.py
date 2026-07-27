# cleaning.py
# Cleans the merged dataset before further processing.
# Input: data/interim/merged_raw.csv
# Output: data/interim/cleaned.csv

import pandas as pd
from . import config


def load_merged():
  # load the dataset produced by data_acquisition.py
  return pd.read_csv(config.MERGED_PATH)


def handle_missing_values(df):
  # drop rows with a missing query or missing label
  before = len(df)
  df = df.dropna(subset=["Sentence", "Label"])
  print(f"Dropped {before - len(df):,} rows with missing values")
  return df


def fix_label_encoding(df):
  # convert labels to numbers, turning invalid values into NaN
  df = df.copy()
  df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
   # drop rows where the label failed to convert
  before = len(df)
  df = df.dropna(subset=["Label"])
  print(f"Dropped {before - len(df):,} rows with invalid label values")

  # cast to integer now that no missing values remain
  df["Label"] = df["Label"].astype(int)
  return df


def standardize_formatting(df):
  # ensure query text is string type
  df = df.copy()
  df["Sentence"] = df["Sentence"].astype(str)

  # remove leading and trailing whitespace
  df["Sentence"] = df["Sentence"].str.strip()

  # collapse repeated inner whitespace into a single space
  df["Sentence"] = df["Sentence"].str.replace(r"\s+", " ", regex=True)

  # drop rows left empty after cleaning
  before = len(df)
  df = df[df["Sentence"].str.len() > 0]
  print(f"Dropped {before - len(df):,} rows with empty text")
  return df


def remove_conflicting_labels(df):
  # count how many distinct labels each query appears under
  label_counts = df.groupby("Sentence")["Label"].nunique()

  # identify queries that appear under more than one label
  conflicting_queries = label_counts[label_counts > 1].index

  # drop all rows for those queries, since the correct label is ambiguous
  before = len(df)
  df = df[~df["Sentence"].isin(conflicting_queries)]
  print(f"Dropped {before - len(df):,} rows with conflicting labels")
  return df


def remove_duplicates(df):
  # remove exact duplicate rows, keeping the first occurrence
  before = len(df)
  df = df.drop_duplicates(subset=["Sentence"])
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