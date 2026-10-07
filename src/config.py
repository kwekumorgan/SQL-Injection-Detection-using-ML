#imports
import os

# FILE PATH

# Get the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


#Core directory structure for data, models, and evaluation outputs
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
INTERIM_DATA_DIR = os.path.join(BASE_DIR, "data", "interim")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OBFUSCATED_DATA_DIR = os.path.join(BASE_DIR, "data", "obfuscated") 

MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")

# RAW DATASETS
SQLIV3_RAW_PATH = os.path.join(RAW_DATA_DIR, "SQLiV3.csv")
DATA_SHEET_1_PATH = os.path.join(RAW_DATA_DIR, "data_sheet_1.csv")


# INTERIM OUTPUT
NATIVE_OBFUSCATED_PATH = os.path.join(INTERIM_DATA_DIR, "sqliv3_native_obfuscated.csv")
MERGED_PATH = os.path.join(INTERIM_DATA_DIR, "merged_raw.csv")
CLEANED_PATH = os.path.join(INTERIM_DATA_DIR, "cleaned.csv") 
TAGGED_PATH = os.path.join(INTERIM_DATA_DIR, "tagged.csv")

BCCC_OBFUSCATED_PATH = os.path.join(OBFUSCATED_DATA_DIR, "bccc_obfuscated.csv")

# PROCESSED OUTPUT
PREPROCESSED_PATH = os.path.join(PROCESSED_DATA_DIR, "preprocessed.csv")
TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train.csv")
TEST_CLEAN_PATH = os.path.join(PROCESSED_DATA_DIR, "test_clean.csv")

TEST_OBFUSCATED_PATH = NATIVE_OBFUSCATED_PATH

VALIDATION_PATH = os.path.join(PROCESSED_DATA_DIR, "validation.csv")
RANDOMIZED_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train_randomized.csv")
RESEARCH_OBFUSCATED_PATH = os.path.join(OBFUSCATED_DATA_DIR, "research_obfuscated.csv")

# SQL comment-based obfuscation (sqlmap: space2comment, randomcomments)
SQLMAP_COMMENT_OBFUSCATED_PATH = os.path.join(OBFUSCATED_DATA_DIR, "sqlmap_comment_obfuscated.csv")

# Case variation obfuscation (sqlmap: randomcase)
SQLMAP_CASE_OBFUSCATED_PATH = os.path.join(OBFUSCATED_DATA_DIR, "sqlmap_case_obfuscated.csv")

SQLMAP_COMBINED_OBFUSCATED_PATH = os.path.join(OBFUSCATED_DATA_DIR, "sqlmap_combined_obfuscated.csv")
SQLMAP_CASE_OBFUSCATED_FULL_PATH = os.path.join(OBFUSCATED_DATA_DIR, "sqlmap_case_obfuscated_full.csv")
SQLMAP_COMMENT_OBFUSCATED_FULL_PATH = os.path.join(OBFUSCATED_DATA_DIR, "sqlmap_comment_obfuscated_full.csv")

# HYPERPARAMETERS
RANDOM_STATE = 2

# Test size
TEST_SIZE = 0.2
BCCC_SAMPLE_SIZE = 2500 # skeleton-deduplication sample, not full pool

# TF-IDF: word-level with symbol tokens preserved(eg. '=', '--')
TFIDF_MAX_FEATURES = None
TFIDF_TOKEN_PATTERN = (
    r"--|/\*|\*/|!=|<>|>=|<=|\|\||\w+|"
    r"@[a-zA-Z_0-9@]*|\[.*?\]|[^\w\s]"
)

TFIDF_NGRAM_RANGE = (1, 1)

# SQL INJECTION ATTACK CHARACTERISTICS
ATTACK_TYPE_PATTERNS = {
    "Tautology": (
        r"(?i)"
        r"(?:"
        r"\b\d+\s*=\s*\d+\b"
        r"|'\s*\w*\s*'\s*=\s*'\s*\w*\s*'"
        r"|\b(?:or|and)\s+\d+\s*=\s*\d+"
        r")"
    ),
    "Boolean": (
        r"(?i)"
        r"\b(?:and|or)\b\s+"
        r"[\w'\"()]+"
        r"\s*(?:=|!=|<>|<|>|<=|>=|like|between)"
    ),
    "Union": (
        r"(?i)"
        r"\bunion\b"
        r"(?:\s+all)?"
        r"\s+\bselect\b"
    ),
    "Time": (
        r"(?i)"
        r"(?:"
        r"\bsleep\s*\("
        r"|\bbenchmark\s*\("
        r"|\bpg_sleep\s*\("
        r"|\bwaitfor\s+delay\b"
        r")"
    ),
    "Error": (
        r"(?i)"
        r"(?:"
        r"\bextractvalue\s*\("
        r"|\bupdatexml\s*\("
        r"|\bfloor\s*\(\s*rand"
        r"|\butl_inaddr\b"
        r"|\bxmltype\s*\("
        r")"
    ),
    "Stacked": (
        r"(?is)"
        r";\s*"
        r"(?:"
        r"select"
        r"|insert"
        r"|update"
        r"|delete"
        r"|drop"
        r"|alter"
        r"|exec"
        r")\b"
    ),
}

# OBFUSCATION INDICATORS

# Hexadecimal representation
HEX_PATTERN = r"\b0x[0-9a-fA-F]+\b"

# URL-encoded characters
URL_ENCODING_PATTERN = r"%[0-9a-fA-F]{2}"

# SQL CHAR()/CHR() representation
CHAR_FUNCTION_PATTERN = (
    r"(?i:\b(?:char|chr)\s*\(\s*"
    r"\d+(?:\s*,\s*\d+)*"
    r"\s*\))"
)

# Comments inserted inside SQL keywords or tokens, e.g. SEL/**/ECT or UN/**/ION
COMMENT_OBFUSCATION_PATTERN = r"\b\w+/\*[\s\S]*?\*/\w+\b"

# Native obfuscation indicators used to exclude obfuscated queries
NATIVE_OBFUSCATION_PATTERN = "|".join([
    HEX_PATTERN,
    URL_ENCODING_PATTERN,
    CHAR_FUNCTION_PATTERN,
    COMMENT_OBFUSCATION_PATTERN,
])