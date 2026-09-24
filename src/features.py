# features.py
# Constructs TF-IDF feature matrices for training and evaluation datasets.

# imports
from sklearn.feature_extraction.text import TfidfVectorizer
from . import config


# Initialize TF-IDF vectorizer preserving SQL operators and key symbols
def build_vectorizer():
    return TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        token_pattern=config.TFIDF_TOKEN_PATTERN,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        lowercase=False,
    )


# Fit vectorizer vocabulary on training text and return transformed matrix
def fit_transform_train(vectorizer, train_df, text_col="Query"):
    return vectorizer.fit_transform(train_df[text_col])


# Transform validation or test text using the pre-fitted vectorizer
def transform_data(vectorizer, data_df, text_col="Query"):
    return vectorizer.transform(data_df[text_col])