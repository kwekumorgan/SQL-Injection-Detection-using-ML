# preprocessing.py
# Prepares cleaned query text for feature extraction.
# Input: data/interim/cleaned.csv
# Output: data/processed/preprocessed.csv



import pandas as pd
from . import config


def load_cleaned():
  # load the dataset produced by cleaning.py
  return pd.read_csv(config.CLEANED_PATH)


def lowercase_text(df):
  # convert all query text to lowercase
  df = df.copy()
  df["Query"] = df["Query"].astype(str).str.lower()
  return df

def clean_non_sql_symbols(df):
  """
  Removes non-SQL junk/symbols (e.g. emojis, stray bullets, control chars)
  WHILE PRESERVING SQL operators and punctuation (-- , /* , */ , != , < , > , = , ' , " , parentheses).
  """
  df = df.copy()

  # Allow letters, digits, whitespace, and SQL-relevant symbols:
  # Everything else (emojis, stray unicode, control chars) gets replaced with a space.
  df["Query"] = df["Query"].astype(str).str.replace(
      r"[^\w\s\-\/\*\!\<\>\=\'\"\(\)\,\;\.\+\%\|]", " ", regex=True
  )

  # Collapse multiple spaces into a single space and strip edges
  df["Query"] = df["Query"].str.replace(r"\s+", " ", regex=True).str.strip()
  return df

def drop_empty_after_cleaning(df):
  # symbol-stripping can reduce some rows (e.g. pure emoji/symbol text) to
  # an empty string, which pandas reads back as NaN on the next CSV load
  before = len(df)
  df = df[df["Query"].str.len() > 0]
  print(f"Dropped {before - len(df):,} rows left empty after symbol cleaning")
  return df



def preprocess_dataset():
  # run the preprocessing pipeline and save the result
  df = load_cleaned()
  print(f"Loaded {len(df):,} rows from cleaned dataset")

  df = lowercase_text(df)
  df = clean_non_sql_symbols(df)
  df = drop_empty_after_cleaning(df)
  df = df.reset_index(drop=True)

  df.to_csv(config.PREPROCESSED_PATH, index=False)
  print(f"Saved {len(df):,} preprocessed rows (non-SQL symbols removed, SQL operators preserved)")
  return df


if __name__ == "__main__":
  preprocess_dataset()