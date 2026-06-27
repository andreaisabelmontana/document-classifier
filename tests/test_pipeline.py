"""The TF-IDF pipeline trains and predicts an easy example correctly."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from docclf import build_linear_pipeline


def test_pipeline_is_wired():
    """The production pipeline has the expected TF-IDF -> classifier shape."""
    pipe = build_linear_pipeline()
    assert isinstance(pipe, Pipeline)
    assert "tfidf" in pipe.named_steps
    assert "clf" in pipe.named_steps
    assert pipe.named_steps["tfidf"].stop_words == "english"


def test_pipeline_trains_and_predicts_easy_example():
    """A TF-IDF + LogisticRegression pipeline learns a separable toy task.

    Uses min_df=1 because the toy corpus is far smaller than 20 Newsgroups; the
    production pipeline's min_df=2 would prune this tiny vocabulary to nothing.
    The point here is to prove TF-IDF features + a linear model classify an
    obviously-separable example correctly.
    """
    docs = [
        "the cat sat on the mat and purred softly",
        "a kitten chased the ball of yarn all day",
        "cats love to nap in the warm afternoon sun",
        "the stock market rallied as investors bought shares",
        "the company reported strong quarterly earnings and profit",
        "bond yields fell while equity prices climbed higher",
    ]
    labels = [0, 0, 0, 1, 1, 1]  # 0=cats, 1=finance

    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", min_df=1)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipe.fit(docs, labels)

    assert pipe.predict(["a fluffy cat napping in the sun"])[0] == 0
    assert pipe.predict(["investors bought shares as profit rose"])[0] == 1

    # predict_proba should be available and well-formed.
    proba = pipe.predict_proba(["a cat chased some yarn"])[0]
    assert abs(proba.sum() - 1.0) < 1e-9
