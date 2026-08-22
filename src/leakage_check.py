# leakage_check.py
# Checks data separation between training and evaluation datasets.
# Exact duplicate overlap should be zero. Structural query-template
# overlap is reported as a diagnostic and is not treated as leakage.

import pandas as pd
from .data_acquisition import _skeleton


# Check exact query overlap and structural query-template overlap
# between training and test sets.
def check_leakage(train_df, test_df, text_col="Query"):

    #CHECK EXACT DUPLICATION OVERLAP
  
    train_queries = set(
        train_df[text_col].astype(str)
    )

    test_queries = test_df[text_col].astype(str)

    exact_overlap = test_queries.isin(train_queries)
    exact_count = exact_overlap.sum()
    exact_pct = exact_overlap.mean() * 100

    print(f"Test rows: {len(test_df):,}")
    print(
        f"Exact query overlap: "
        f"{exact_count:,} ({exact_pct:.2f}%)"
    )

    #CHECK STRUCTURAL OVERLAP AS A DIAGNOSTIC
  
   

    train_templates = set(
        train_df[text_col].apply(_skeleton)
    )

    test_templates = test_df[text_col].apply(_skeleton)

    structural_overlap = test_templates.isin(train_templates)
    structural_count = structural_overlap.sum()
    structural_pct = structural_overlap.mean() * 100

    print(
        f"Structural template overlap: "
        f"{structural_count:,} ({structural_pct:.2f}%)"
    )

    # REPORT FINDINGS

    if exact_count > 0:
        print(
            "WARNING: exact query overlap detected -- "
            "check the train/test split."
        )
    else:
        print("No exact query overlap detected.")

    print(
        "Structural overlap is reported as a diagnostic only "
        "and is not treated as data leakage."
    )

    # Exact duplicates are the condition treated as leakage
    # for this dataset partition check.
    return exact_count == 0


# Execute leakage validation check directly against processed datasets.
if __name__ == "__main__":
    from . import config

    # 1. Load exported train and clean test datasets
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_CLEAN_PATH)

    # 2. Run leakage validation check
    check_leakage(train_df, test_df)