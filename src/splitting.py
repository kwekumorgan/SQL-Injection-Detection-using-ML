# splitting.py
# Prepares model inputs by splitting preprocessed data into training and clean test sets.
# Output: data/processed/train.csv, data/processed/test_clean.csv

import pandas as pd
from sklearn.model_selection import train_test_split
from . import config
from .data_acquisition import _skeleton


# Load preprocessed corpus from disk.
def load_preprocessed():
    return pd.read_csv(config.PREPROCESSED_PATH)


# Assign structural skeleton templates to group near-identical query structures.
def add_skeleton_groups(df):
    df = df.copy()
    
    # Generate structural templates for every query sentence
    df["_skeleton"] = df["Sentence"].apply(_skeleton)
    return df


# Perform group-wise split to ensure no single skeleton structure spans across train and test.
def split_dataset(df):
    # Extract unique structural group signatures
    groups = df["_skeleton"].unique()

    # Split groups rather than individual rows to eliminate data leakage
    train_groups, test_groups = train_test_split(
        groups,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    # Filter original dataframe based on assigned group subsets
    train_df = df[df["_skeleton"].isin(train_groups)]
    test_df = df[df["_skeleton"].isin(test_groups)]

    # Remove temporary grouping metadata column
    return train_df.drop(columns="_skeleton"), test_df.drop(columns="_skeleton")


# Execute complete splitting workflow and export train/test CSVs to disk.
def run_split():
    # 1. Load preprocessed dataset
    df = load_preprocessed()
    print(f"Loaded {len(df):,} rows from preprocessed dataset")

    # 2. Attach skeleton metadata tags
    df = add_skeleton_groups(df)

    # 3. Perform group-based train/test partition
    train_df, test_df = split_dataset(df)

    # 4. Clean indices
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # 5. Log dataset split statistics
    print(f"Training set: {len(train_df):,} rows")
    print(f"Test set:     {len(test_df):,} rows")

    # 6. Export split sets to processed directory
    train_df.to_csv(config.TRAIN_PATH, index=False)
    test_df.to_csv(config.TEST_CLEAN_PATH, index=False)
    
    return train_df, test_df


# Run script independently from command line
if __name__ == "__main__":
    run_split()