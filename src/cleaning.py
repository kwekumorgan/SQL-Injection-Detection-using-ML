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