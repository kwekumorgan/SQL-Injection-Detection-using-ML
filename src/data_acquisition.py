# # data_acquisition.py
# Loads raw data sources, extracts skeleton-deduplicated BCCC subsets, and merges datasets.
# Output destination: data/interim/merged_raw.csv
import re
import pandas as pd

from . import config

def load_sqliv3():
  #load SQLiV3 corpus and isolate required feature schema

  df = pd.read_csv(config.SQLIV3_RAW_PATH, encoding="utf-8")

  #drop artifact indexing columns from raw dataset export
  df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
  df=df.rename(columns={"Sentence": "Sentence", "Label": "Label"})
  df["source"]="sqliv3"
  return df[["Sentence", "Label", "source"]]


def separate_native_obfuscation(df, text_col="Sentence"):
    
    # Detect natively obfuscated rows using regex
    is_obfuscated = df[text_col].astype(str).str.contains(
        config.NATIVE_OBFUSCATION_PATTERN, regex=True, na=False
    )
    
    # Split into separate dataframes
    obfuscated_rows = df[is_obfuscated].copy().reset_index(drop=True)
    clean_rows = df[~is_obfuscated].copy().reset_index(drop=True)

    # Reporting
    print(f"SQLiV3 total: {len(df):,}")
    print(f"  Natively obfuscated (extracted): {len(obfuscated_rows):,}")
    print(f"  Clean (kept for baseline pool):  {len(clean_rows):,}")

    return clean_rows, obfuscated_rows



def _skeleton(text):
  #Abstract query variations by mapping absolute numeric constants to a standard token 
  return re.sub(r"\d+", "N", str(text))

def dedupe_by_skeleton(df,text_col, sample_size, random_state):
  #Filter duplicate syntactic structures and sample remaining rows to optimize performance
  df=df.copy()

  #Isolate structural templates to filter out literal variance 
  df["_skeleton"] = df[text_col].apply(_skeleton)
  deduped = df.drop_duplicates(subset="_skeleton")

  # Enforce sampling bounds to optimize training distribution balance
  if len(deduped) > sample_size:
    deduped = deduped.sample(n=sample_size, random_state=random_state)

  return deduped.drop(columns="_skeleton").reset_index(drop=True)


def load_bccc_error_based():
  #load BCCC, isloate error-based rows, and reduce to a diverse sample
  df = pd.read_csv(config.BCCC_RAW_PATH)
  df = df.rename(columns = {"Data": "Sentence"})


  is_error_based = df["Sentence"].str.contains(
    config.BCCC_ERROR_PATTERN, regex=True , na=False
  )

  df = df[is_error_based].copy()
  df["Label"] = 1
  df["source"] = "bccc"

  df = dedupe_by_skeleton(
    df, text_col="Sentence",
    sample_size=config.BCCC_SAMPLE_SIZE,
    random_state=config.RANDOM_STATE,

  )
  return df[["Sentence", "Label", "source"]]


def build_merged_dataset():
  #load, separate obfuscation, merge clean pools, persist all outputs
  sqliv3_df = load_sqliv3()
  sqliv3_clean, sqliv3_obfuscated = separate_native_obfuscation(sqliv3_df)
  bccc_df = load_bccc_error_based()

  merged = pd.concat([sqliv3_clean, bccc_df], ignore_index=True)

  print(f"BCCC error-based rows added: {len(bccc_df):,}")
  print(f"Merged (clean baseline pool) total: {len(merged):,}")

  sqliv3_obfuscated.to_csv(config.NATIVE_OBFUSCATED_PATH, index=False)
  merged.to_csv(config.MERGED_PATH, index=False)
  return merged, sqliv3_obfuscated