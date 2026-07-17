#imports
import os

# File paths 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
INTERIM_DATA_DIR = os.path.join(BASE_DIR, "data", "interim")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

MODELS_DIR= os.path.join(BASE_DIR, "models")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")

SQLIV3_RAW_PATH = os.path.join(RAW_DATA_DIR, "SQLiV3.csv")
BCCC_RAW_PATH = os.path.join(RAW_DATA_DIR, "BCCC-SFU-SQLInj-2023.csv")


MERGED_PATH = os.path.join(INTERIM_DATA_DIR, "merged_raw.csv")
CLEANED_PATH = os.path.join(INTERIM_DATA_DIR, "cleaned.csv") 
TAGGED_PATH = os.path.join(INTERIM_DATA_DIR, "tagged.csv")

NORMALIZED_PATH = os.path.join(PROCESSED_DATA_DIR, "normalized.csv")
TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train.csv")
TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "test.csv")