# features.py
# Constructs TF-IDF feature matrices for training and evaluation datasets.

from sklearn.feature_extraction.text import TfidfVectorizer
from . import config


# Initialize TF-IDF vectorizer preserving SQL operators and key symbols.
def build_vectorizer():
    return TfidfVectorizer(
        max_features=None,
        token_pattern=config.TFIDF_TOKEN_PATTERN,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        lowercase=True,
    )


# Fit vectorizer vocabulary on training text and return transformed matrix.
def fit_transform_train(vectorizer, train_df, text_col="Sentence"):
    return vectorizer.fit_transform(train_df[text_col])


# Transform evaluation text using the pre-fitted vectorizer (no re-fitting).
def transform_test(vectorizer, test_df, text_col="Sentence"):
    return vectorizer.transform(test_df[text_col])