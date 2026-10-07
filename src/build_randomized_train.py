# build_randomized_train.py


import pandas as pd

from . import config
from .number_randomization import apply_randomization


def build_randomized_train():
    train_df = pd.read_csv(config.TRAIN_PATH)
    print(f"Loaded {len(train_df):,} rows from {config.TRAIN_PATH}")

    train_randomized = apply_randomization(
        train_df,
        seed_malicious=config.RANDOM_STATE,
        seed_benign=config.RANDOM_STATE + 1,
    )

    train_randomized.to_csv(config.RANDOMIZED_TRAIN_PATH, index=False)
    print(f"Saved randomized training set -> {config.RANDOMIZED_TRAIN_PATH}")

    return train_randomized


if __name__ == "__main__":
    build_randomized_train()