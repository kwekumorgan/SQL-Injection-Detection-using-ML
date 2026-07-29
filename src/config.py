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
RBSQLI_RAW_PATH = os.path.join(RAW_DATA_DIR, "RbSQLi.csv")
RBSQLI_CATEGORY_TARGETS = {
    "error": 3500,
    "time": 2300,
    "union": 1400,
}

#interim output
NATIVE_OBFUSCATED_PATH = os.path.join(INTERIM_DATA_DIR, "sqliv3_native_obfuscated.csv")
MERGED_PATH = os.path.join(INTERIM_DATA_DIR, "merged_raw.csv")
CLEANED_PATH = os.path.join(INTERIM_DATA_DIR, "cleaned.csv") 
TAGGED_PATH = os.path.join(INTERIM_DATA_DIR, "tagged.csv")


#processed output 
PREPROCESSED_PATH = os.path.join(PROCESSED_DATA_DIR, "preprocessed.csv")
TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train.csv")
TEST_CLEAN_PATH = os.path.join(PROCESSED_DATA_DIR, "test_clean.csv")
TEST_OBFUSCATED_PATH = os.path.join(PROCESSED_DATA_DIR, "test_obfuscated.csv")



# HYPERPARAMETERS

RANDOM_STATE = 42

#TF-IDF: word-level with symbol tokens preserved(eg. '=', '--')
TFIDF_MAX_FEATURES= 5000
TFIDF_TOKEN_PATTERN = r"\w+|[^\w\s]"
TFIDF_NGRAM_RANGE = (1,3)

TEST_SIZE = 0.2
BCCC_SAMPLE_SIZE = 2500 # skeleton-deduplication sample, not full pool

# feature selection — k-search candidates
FEATURE_SELECTION_K_CANDIDATES = [100, 500, 1000, 2000]


# OBFUSCATION DETECTION PATTERNS

HEX_PATTERN = r"0x[0-9a-fA-F]+"
URL_ENCODING_PATTERN = r"%[0-9a-fA-F]{2}"
CHAR_FUNCTION_PATTERN = r"\b(?i:cha?r)\s*\(\s*\d+(?:\s*,\s*\d+)*\s*\)"
BLOCK_COMMENT_PATTERN = r"/\*[\s\S]*?\*/"

#case-insensitivity to keywords only, not whole pattern
MIXED_CASE_PATTERN = (
    r"\b(?=\w*[a-z])(?=\w*[A-Z])"
    r"(?i:select|union|insert|update|delete|drop|and|or|where|from)\b"
)


# combined pattern for the SQLiV3 separation step
NATIVE_OBFUSCATION_PATTERN = "|".join([
    HEX_PATTERN, URL_ENCODING_PATTERN, CHAR_FUNCTION_PATTERN,
    BLOCK_COMMENT_PATTERN, MIXED_CASE_PATTERN,
])


# BCCC-specific: used only to filter BCCC down to error-based rows
BCCC_ERROR_PATTERN = r"(?i)(extractvalue|updatexml|floor\s*\(\s*rand|utl_inaddr|xmltype)"


#ATTACK_TYPE PATTERN 

ATTACK_TYPE_PATTERNS = {
    "Tautology": r"(?i)(\d+\s*=\s*\d+|'\w+'\s*=\s*'\w+'|or\s+\d+\s*=\s*\d+)",
    "Comment": r"(--|#|/\*[\s\S]*?\*/)",
    "Boolean": r"(?i)\b(and|or)\b\s+[\w'\"]+\s*(=|<|>|like|between)",
    "Union": r"(?i)\bunion\b",
    "Time": r"(?i)(sleep\s*\(|waitfor\s+delay|pg_sleep\s*\(|benchmark\s*\()",
    # Includes Oracle xmltype/utl_inaddr in addition to the original
    # MySQL-only definition (extractvalue/updatexml/floor-rand), broadened
    # to cover the BCCC source.
    "Error": r"(?i)(extractvalue|updatexml|floor\s*\(\s*rand|utl_inaddr|xmltype)",
    "Stacked": r";.{1,}(select|insert|update|delete|drop)",
}