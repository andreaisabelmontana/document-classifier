"""The from-scratch Multinomial NB must agree with sklearn's on a toy corpus."""

import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB as SklearnMNB

from docclf import MultinomialNaiveBayes

# A tiny two-class corpus: "sports" vs "tech".
TOY_DOCS = [
    "the team won the game and scored a goal",
    "our team lost the match but scored late",
    "the players ran across the field at the game",
    "the new laptop has a fast cpu and lots of memory",
    "this phone ships with a powerful chip and a big screen",
    "the server crashed because the memory and cpu ran out",
]
TOY_LABELS = np.array([0, 0, 0, 1, 1, 1])  # 0=sports, 1=tech


def _toy_counts():
    vec = CountVectorizer()
    X = vec.fit_transform(TOY_DOCS)
    return X, TOY_LABELS


@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.1])
def test_matches_sklearn_log_prob(alpha):
    """Feature log-probabilities and priors must match sklearn within tol."""
    X, y = _toy_counts()

    mine = MultinomialNaiveBayes(alpha=alpha).fit(X, y)
    ref = SklearnMNB(alpha=alpha).fit(X, y)

    np.testing.assert_allclose(
        mine.feature_log_prob_, ref.feature_log_prob_, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        mine.class_log_prior_, ref.class_log_prior_, rtol=1e-10, atol=1e-10
    )


@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.1])
def test_matches_sklearn_predictions_and_proba(alpha):
    """Predictions and posterior probabilities must match sklearn within tol."""
    X, y = _toy_counts()

    mine = MultinomialNaiveBayes(alpha=alpha).fit(X, y)
    ref = SklearnMNB(alpha=alpha).fit(X, y)

    # Same training docs are fine to score for an equivalence check.
    np.testing.assert_array_equal(mine.predict(X), ref.predict(X))
    np.testing.assert_allclose(
        mine.predict_proba(X), ref.predict_proba(X), rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(
        mine.predict_log_proba(X), ref.predict_log_proba(X), rtol=1e-8, atol=1e-8
    )


def test_learns_toy_corpus():
    """A held-out sentence is classified into the obviously-correct class."""
    vec = CountVectorizer().fit(TOY_DOCS)
    X = vec.transform(TOY_DOCS)
    nb = MultinomialNaiveBayes(alpha=1.0).fit(X, TOY_LABELS)

    sports_like = vec.transform(["the team scored a goal in the game"])
    tech_like = vec.transform(["the laptop cpu and memory are fast"])

    assert nb.predict(sports_like)[0] == 0
    assert nb.predict(tech_like)[0] == 1


def test_proba_rows_sum_to_one():
    X, y = _toy_counts()
    nb = MultinomialNaiveBayes().fit(X, y)
    proba = nb.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(proba.shape[0]), atol=1e-10)
