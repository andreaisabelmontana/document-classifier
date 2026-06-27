"""Dataset loading and the TF-IDF + linear-model pipeline.

Keeps the data and model wiring in one place so ``train.py``, ``predict.py``
and the tests all agree on exactly which categories, vectoriser settings and
estimator are in play.
"""

from __future__ import annotations

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# A focused 5-category slice of 20 Newsgroups. Chosen to span clearly
# different topics (autos, hardware, medicine, space, politics) so the task is
# meaningful but still fast to train on a laptop.
CATEGORIES = [
    "rec.autos",
    "comp.sys.mac.hardware",
    "sci.med",
    "sci.space",
    "talk.politics.guns",
]


def load_dataset(subset: str, *, remove_metadata: bool = True):
    """Fetch a subset ("train" or "test") of the chosen 20 Newsgroups slice.

    ``remove_metadata=True`` strips headers, footers and quoted text so the
    classifier learns from the actual prose rather than giveaway artefacts
    like the newsgroup name in the header. This makes the numbers honest (and
    a bit lower) compared to leaving that leakage in.
    """
    remove = ("headers", "footers", "quotes") if remove_metadata else ()
    return fetch_20newsgroups(
        subset=subset,
        categories=CATEGORIES,
        remove=remove,
        shuffle=True,
        random_state=42,
    )


def target_names():
    """Human-readable class names, in label order, for the chosen slice."""
    # fetch_20newsgroups returns categories sorted; mirror that here.
    return sorted(CATEGORIES)


def build_linear_pipeline() -> Pipeline:
    """TF-IDF vectoriser + multinomial logistic-regression classifier.

    The vectoriser lowercases, removes English stopwords, keeps uni- and
    bi-grams, and drops terms that are too rare or too common. The classifier
    is a plain L2-regularised logistic regression — fast, strong, and its
    weights stay interpretable.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=10.0,
                    max_iter=1000,
                ),
            ),
        ]
    )
