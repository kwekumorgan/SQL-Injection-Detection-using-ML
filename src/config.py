#imports
import os

# File paths 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
INTERIM_DATA_DIR = os.path.join(BASE_DIR, "data", "interim")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

MODELS_DIR= os.path.join(BASE_DIR, "models")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")

#RAW DATASETS
SQLIV3_RAW_PATH = os.path.join(RAW_DATA_DIR, "SQLiV3.csv")
DATA_SHEET_1_PATH = os.path.join(RAW_DATA_DIR, "data_sheet_1.csv")


#INTERIM OUTPUT
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
TEST_SIZE = 0.2
BCCC_SAMPLE_SIZE = 2500 # skeleton-deduplication sample, not full pool

#TF-IDF: word-level with symbol tokens preserved(eg. '=', '--')
TFIDF_MAX_FEATURES= None
TFIDF_TOKEN_PATTERN =  (
    r"--|/\*|\*/|!=|<>|>=|<=|\|\||\w+|"
    r"@[a-zA-Z_0-9@]*|\[.*?\]|[^\w\s]"
)

TFIDF_NGRAM_RANGE = (1,3)






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


#OBFUSCATION INDICATORS 


# Hexadecimal representation
HEX_PATTERN = r"\b0x[0-9a-fA-F]+\b"


# URL-encoded characters
URL_ENCODING_PATTERN = r"%[0-9a-fA-F]{2}"


# SQL CHAR() representation
CHAR_FUNCTION_PATTERN = (
    r"(?i)"
    r"\bchar\s*\(\s*"
    r"\d+(?:\s*,\s*\d+)*"
    r"\s*\)"
)


# SQL comments
LINE_COMMENT_PATTERN = r"(?:--|#)(?:\s|$)"

BLOCK_COMMENT_PATTERN = r"/\*[\s\S]*?\*/"

COMMENT_PATTERN = (
    f"(?:{LINE_COMMENT_PATTERN}|{BLOCK_COMMENT_PATTERN})"
)

NATIVE_OBFUSCATION_PATTERN = "|".join([
    HEX_PATTERN,
    URL_ENCODING_PATTERN,
    CHAR_FUNCTION_PATTERN,
])



