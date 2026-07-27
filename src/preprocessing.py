# preprocessing.py
# Prepares cleaned query text for feature extraction.
# Input: data/interim/cleaned.csv
# Output: data/processed/preprocessed.csv
#


import pandas as pd
from . import config


def load_cleaned():
  # load the dataset produced by cleaning.py
  return pd.read_csv(config.CLEANED_PATH)


def lowercase_text(df):
  # convert all query text to lowercase
  df = df.copy()
  df["Sentence"] = df["Sentence"].astype(str).str.lower()
  return df


def preprocess_dataset():
  # run the preprocessing pipeline and save the result
  df = load_cleaned()
  print(f"Loaded {len(df):,} rows from cleaned dataset")

  df = lowercase_text(df)

  df.to_csv(config.PREPROCESSED_PATH, index=False)
  print(f"Saved {len(df):,} preprocessed rows")
  return df


if __name__ == "__main__":
  preprocess_dataset()