# leakage_check.py
# Validates data separation between training and evaluation datasets.
# Ensures zero structural query template overlap (data leakage) prior to model training.

import pandas as pd
from .data_acquisition import _skeleton


# Calculate structural overlap between training and test set query skeletons.
def check_leakage(train_df, test_df, text_col="Sentence"):
    # Generate structural templates for training and test sets
    train_templates = set(train_df[text_col].apply(_skeleton))
    test_templates = test_df[text_col].apply(_skeleton)

    # Detect test set templates that exist in the training pool
    leaked = test_templates.isin(train_templates)
    leak_count = leaked.sum()
    leak_pct = leaked.mean() * 100

    # Log verification findings
    print(f"Test rows: {len(test_df):,}")
    print(f"Leaked (template also in training): {leak_count:,} ({leak_pct:.2f}%)")

    # Flag potential dataset integrity issues
    if leak_count > 0:
        print("WARNING: leakage detected -- check the split logic before proceeding")
    else:
        print("No leakage detected")

    # Return True if dataset partition is completely leak-free
    return leak_count == 0


# Execute leakage validation check directly against processed datasets
if __name__ == "__main__":
    from . import config
    
    # 1. Load exported train and clean test datasets
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_CLEAN_PATH)
    
    # 2. Run structural leakage verification
    check_leakage(train_df, test_df)