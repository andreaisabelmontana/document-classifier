"""docclf: a small, end-to-end document-classification toolkit.

Exposes a from-scratch Multinomial Naive Bayes classifier and helpers for
building a TF-IDF + linear-model pipeline on the 20 Newsgroups corpus.
"""

from .naive_bayes import MultinomialNaiveBayes
from .pipeline import (
    CATEGORIES,
    build_linear_pipeline,
    load_dataset,
    target_names,
)

__all__ = [
    "MultinomialNaiveBayes",
    "CATEGORIES",
    "build_linear_pipeline",
    "load_dataset",
    "target_names",
]
