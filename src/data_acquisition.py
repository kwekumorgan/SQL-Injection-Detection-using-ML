#data_acquisition.py
# Loads raw data sources, extracts skeleton-deduplicated subsets, and merges datasets.
import re
import pandas as pd
from . import config



# Normalize query text into a generic template for structural deduplication.
def _skeleton(text):
    text = str(text).lower()                             # Convert to lowercase for uniform comparison
    text = re.sub(r"'[^']*'", "'STR'", text)             # Replace single-quoted string literals with 'STR'
    text = re.sub(r'"[^"]*"', '"STR"', text)             # Replace double-quoted string literals with 'STR'
    text = re.sub(r"\b\d+\b", "N", text)                  # Replace standalone numbers with 'N' (preserves function names)
    text = re.sub(r"\s+", " ", text).strip()             # Collapse multiple spaces into a single space
    return text



# Group rows by structural skeleton and sample to enforce diversity and limit redundancy.
def dedupe_by_skeleton(df, text_col, sample_size, random_state, max_per_skeleton=3):
    df = df.copy()
    
    # Generate structural templates for every query
    df["_skeleton"] = df[text_col].apply(_skeleton)

    # Sample up to max_per_skeleton rows per unique structural pattern
    sampled_indices = (
        df.groupby("_skeleton", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), max_per_skeleton), random_state=random_state))
        .index
    )
    deduped = df.loc[sampled_indices]

    # cap the overall pool size if it still exceeds the target 
    if len(deduped) > sample_size:
        deduped = deduped.sample(n=sample_size, random_state=random_state)

    # Clean up temporary metadata column before returning
    return deduped.drop(columns="_skeleton").reset_index(drop=True)

# Safely coerce Label column to clean integers (0 or 1) and drop invalid rows.
def standardize_labels(df):
    if "Label" in df.columns:
        df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
        df = df.dropna(subset=["Label"])
        df["Label"] = df["Label"].astype(int)
    return df



#load primary SQLiV3 dataset and standardize source schema
def load_sqliv3():
  df = pd.read_csv(config.SQLIV3_RAW_PATH, encoding="utf-8", low_memory=False)

  # Raw file uses "Sentence" as the column header; rename to "Query" for internal use
  df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
  df=df.rename(columns={"Sentence": "Query", "Label": "Label"})
  df["source"]="sqliv3"
  df["attack_type"] = "unknown"
  df = standardize_labels(df)
  
  return df[["Query", "Label", "source","attack_type"]]

def load_sqli_v2_legacy():
    # Load sqli.csv (legacy dataset, often utf-16 encoded)
    try:
        df = pd.read_csv(config.SQLIV_RAW_PATH, encoding="utf-16", low_memory=False)
    except Exception:
        df = pd.read_csv(config.SQLIV_RAW_PATH, encoding="utf-8", low_memory=False)
    
    df = df.rename(columns={"Sentence": "Query", "Label": "Label"})
    df["source"] = "sqli_legacy"
    df["attack_type"] = "unknown"
    return df[["Query", "Label", "source", "attack_type"]]




def load_data_sheet_1():
    # Load supplementary Data Sheet 1 dataset
    df = pd.read_csv(config.DATA_SHEET_1_PATH, encoding="utf-8", low_memory=False)
    df = df.rename(columns={"Sentence": "Query", "Label": "Label"})
    df["source"] = "data_sheet_1"
    df["attack_type"] = "unknown"
    df = standardize_labels(df)
    return df[["Query", "Label", "source", "attack_type"]]



# Split SQLiV3 into clean queries for training and obfuscated queries for test evaluation.
def separate_native_obfuscation(df, text_col="Query"):
    
    # Detect natively obfuscated rows using regex
    is_obfuscated = df[text_col].astype(str).str.contains(
        config.NATIVE_OBFUSCATION_PATTERN, regex=True, na=False
    )
    
    # Separate dataset based on obfuscation presence
    obfuscated_rows = df[is_obfuscated].copy().reset_index(drop=True)
    clean_rows = df[~is_obfuscated].copy().reset_index(drop=True)

    # Log extraction results
    print(f"Merged total: {len(df):,}")
    print(f"  Natively obfuscated (extracted): {len(obfuscated_rows):,}")
    print(f"  Clean (kept for baseline pool):  {len(clean_rows):,}")
    return clean_rows, obfuscated_rows






# Main pipeline execution: process all sources, route pools, and export processed CSVs.
def build_merged_dataset():
    # 1. Load primary corpus and separate clean vs obfuscated rows
    sqliv3_df = load_sqliv3()  
    sheet1_df = load_data_sheet_1()            

     # 2. Combine desired datasets together
    merged_raw = pd.concat([sqliv3_df, sheet1_df], ignore_index=True)
    merged_raw = merged_raw.dropna(subset=["Query"])
    merged_raw["Query"] = merged_raw["Query"].astype(str).str.strip()
    merged_raw = merged_raw[merged_raw["Query"] != "#NAME?"]

    # 3. Standardize to lowercase before obfuscation checks and deduplication
    merged_raw["Query"] = merged_raw["Query"].str.lower()

    # 4. Separate clean vs natively-obfuscated pools
    clean_pool, obfuscated_pool = separate_native_obfuscation(merged_raw)

    # 5. Log pipeline outputs
    print(f"\n--- Data Acquisition Summary ---")
    print(f"Merged total raw samples: {len(merged_raw):,}")
    print(f"Clean baseline training pool total: {len(clean_pool):,}")
    print(f"Obfuscated test evaluation pool total: {len(obfuscated_pool):,}")

    # 6. Export processed datasets to disk
    obfuscated_pool.to_csv(config.NATIVE_OBFUSCATED_PATH, index=False)
    clean_pool.to_csv(config.MERGED_PATH, index=False)

    return clean_pool, obfuscated_pool


# Execute preprocessing pipeline when script is run directly
if __name__ == "__main__":
    build_merged_dataset()