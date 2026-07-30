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
def dedupe_by_skeleton(df, text_col, sample_size, random_state, max_per_skeleton=1):
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



#load primary SQLiV3 dataset and standardize source schema
def load_sqliv3():
  df = pd.read_csv(config.SQLIV3_RAW_PATH, encoding="utf-8")

  # Remove unnamed export index columns
  df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
  df=df.rename(columns={"Sentence": "Sentence", "Label": "Label"})
  df["source"]="sqliv3"
  
  return df[["Sentence", "Label", "source"]]


# Split SQLiV3 into clean queries for training and obfuscated queries for test evaluation.
def separate_native_obfuscation(df, text_col="Sentence"):
    
    # Detect natively obfuscated rows using regex
    is_obfuscated = df[text_col].astype(str).str.contains(
        config.NATIVE_OBFUSCATION_PATTERN, regex=True, na=False
    )
    
    # Separate dataset based on obfuscation presence
    obfuscated_rows = df[is_obfuscated].copy().reset_index(drop=True)
    clean_rows = df[~is_obfuscated].copy().reset_index(drop=True)

    # Log extraction results
    print(f"SQLiV3 total: {len(df):,}")
    print(f"  Natively obfuscated (extracted): {len(obfuscated_rows):,}")
    print(f"  Clean (kept for baseline pool):  {len(clean_rows):,}")

    return clean_rows, obfuscated_rows





# Load BCCC error-based payloads (100% obfuscated, used exclusively for test evaluation).
def load_bccc_error_based(cap_sample=True):
    df = pd.read_csv(config.BCCC_RAW_PATH)
    df = df.rename(columns={"Data": "Sentence"})

    # Filter strictly for error-based payloads using config regex pattern
    is_error_based = df["Sentence"].astype(str).str.contains(
        config.BCCC_ERROR_PATTERN, regex=True, na=False
    )
    df = df[is_error_based].copy()
    
    # Assign metadata labels
    df["Label"] = 1
    df["source"] = "bccc"

    # Determine whether to cap the sample or keep all valid deduplicated rows
    sample_size = config.BCCC_SAMPLE_SIZE if cap_sample else len(df)

    # Deduplicate redundant payloads by template
    df = dedupe_by_skeleton(
        df,
        text_col="Sentence",
        sample_size=sample_size,
        random_state=config.RANDOM_STATE,
        max_per_skeleton=1
    )
    return df[["Sentence", "Label", "source"]]


# Extract, filter, and balance clean SQLi payloads from RbSQLi across key attack types.
def load_rbsqli_supplement():
    df = pd.read_csv(config.RBSQLI_RAW_PATH)
    df = df.rename(columns={"sql_query": "Sentence"})

    # Map attack categories to regex search patterns
    category_patterns = {
        "union": r"(?i)union",
        "time": r"(?i)time",
        "error": r"(?i)error",
    }

    sampled_parts = []
    for name, pattern in category_patterns.items():
        # Isolate payloads belonging to current category
        subset = df[df["injection_type"].astype(str).str.contains(pattern, regex=True, na=False)].copy()

        # Remove obfuscated rows to maintain clean training baseline standards
        is_obfuscated = subset["Sentence"].astype(str).str.contains(
            config.NATIVE_OBFUSCATION_PATTERN, regex=True, na=False
        )
        subset = subset[~is_obfuscated]

        # Assign metadata labels
        subset["Label"] = 1
        subset["source"] = f"rbsqli_{name}"

        # Deduplicate and sample to category target size specified in config
        subset = dedupe_by_skeleton(
            subset,
            text_col="Sentence",
            sample_size=config.RBSQLI_CATEGORY_TARGETS[name],
            random_state=config.RANDOM_STATE,
            max_per_skeleton=1
        )
        
        print(f"RbSQLi {name}: {len(subset):,} clean rows extracted after deduplication")
        sampled_parts.append(subset)

    # Combine sampled subsets into a single DataFrame
    result = pd.concat(sampled_parts, ignore_index=True)
    return result[["Sentence", "Label", "source"]]



# Main pipeline execution: process all sources, route pools, and export processed CSVs.
def build_merged_dataset():
    # 1. Load primary corpus and separate clean vs obfuscated rows
    sqliv3_df = load_sqliv3()
    sqliv3_clean, sqliv3_obfuscated = separate_native_obfuscation(sqliv3_df)

    # 2. Load supplemental sources
    bccc_df = load_bccc_error_based(cap_sample=True)   # Keep full deduplicated test set
    rbsqli_df = load_rbsqli_supplement()               # Clean supplementary training samples

    # 3. Combine pools by purpose
    # Pool A: Obfuscated queries designated for evasion attack evaluation
    obfuscated_test_material = pd.concat([sqliv3_obfuscated, bccc_df], ignore_index=True)
    
    # Pool B: Clean queries for baseline training and validation
    merged = pd.concat([sqliv3_clean, rbsqli_df], ignore_index=True)

    # 4. Log pipeline outputs
    print(f"\n--- Data Acquisition Summary ---")
    print(f"RbSQLi supplementary rows added: {len(rbsqli_df):,}")
    print(f"Clean baseline training pool total: {len(merged):,}")
    print(f"Obfuscated test evaluation pool total: {len(obfuscated_test_material):,}")

    # 5. Export processed datasets to disk
    obfuscated_test_material.to_csv(config.NATIVE_OBFUSCATED_PATH, index=False)
    merged.to_csv(config.MERGED_PATH, index=False)

    return merged, obfuscated_test_material


# Execute preprocessing pipeline when script is run directly
if __name__ == "__main__":
    build_merged_dataset()