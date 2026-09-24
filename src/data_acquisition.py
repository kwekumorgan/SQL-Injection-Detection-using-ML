#data_acquisition.py



# Loads raw data sources and merges datasets.
import pandas as pd
from . import config


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



# Load supplementary Data Sheet 1, keeping both benign and malicious rows.
def load_data_sheet_1():
    df = pd.read_csv(config.DATA_SHEET_1_PATH, encoding="utf-8", low_memory=False)
    df = df.rename(columns={"Sentence": "Query", "Label": "Label"})
    df["source"] = "data_sheet_1"
    df["attack_type"] = "unknown"
    df = standardize_labels(df)

    return df[["Query", "Label", "source", "attack_type"]]




# Split the merged dataset into native-obfuscated queries and plain queries.
def separate_native_obfuscation(df, text_col="Query"):
    
   # Detect native obfuscation indicators using the patterns
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

    #Load primary corpus and separate clean vs obfuscated rows
    sqliv3_df = load_sqliv3()  
    sheet1_df = load_data_sheet_1()            

     #Combine the selected datasets.
    merged_raw = pd.concat([sqliv3_df, sheet1_df], ignore_index=True)
    merged_raw = merged_raw.dropna(subset=["Query"])
    merged_raw["Query"] = merged_raw["Query"].astype(str).str.strip()
    merged_raw = merged_raw[merged_raw["Query"] != "#NAME?"]

    # 3. Standardize to lowercase before obfuscation checks
    merged_raw["Query"] = merged_raw["Query"].str.lower()

    # 4. Separate clean vs natively-obfuscated queries.
    clean_pool, obfuscated_pool = separate_native_obfuscation(merged_raw)

    # 5. Log pipeline outputs
    print(f"\n--- Data Acquisition Summary ---")
    print(f"Merged total raw samples: {len(merged_raw):,}")
    print(f"plain baseline training pool total: {len(clean_pool):,}")
    print(f"Obfuscated test evaluation pool total: {len(obfuscated_pool):,}")
    
    # 6. Export datasets to disk
    obfuscated_pool.to_csv(config.NATIVE_OBFUSCATED_PATH, index=False)
    clean_pool.to_csv(config.MERGED_PATH, index=False)

    return clean_pool, obfuscated_pool


# Execute preprocessing pipeline when script is run directly
if __name__ == "__main__":
    build_merged_dataset()