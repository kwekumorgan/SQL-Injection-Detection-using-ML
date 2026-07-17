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


# HYPERPARAMETERS

RANDOM_STATE = 42

#TF-IDF: word-level with symbol tokens preserved(eg. '=', '--')
TFIDF_MAX_FEATURES= 5000
TFIDF_TOKEN_PATTERN = r"\w+|[^\w\s]"
TFIDF_NGRAM_RANGE = (1,3)

TEST_SIZE = 0.2
BCCC_SAMPLE_SIZE = 2500 # skeleton-deduplication sample, not full pool