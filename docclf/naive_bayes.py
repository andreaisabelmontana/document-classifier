"""From-scratch Multinomial Naive Bayes, implemented in numpy.

This is the genuine algorithm code: no sklearn estimator is used to fit or
predict. It follows the standard multinomial event model with Laplace
(add-alpha) smoothing and works in log space to stay numerically stable.

The maths
---------
For class ``c`` and a document represented as a vector of term counts
``x = (x_1, ..., x_V)``:

    log P(c | x)  ∝  log P(c) + Σ_t  x_t · log P(t | c)

with

    P(t | c) = (N_{tc} + alpha) / (N_c + alpha · V)

where ``N_{tc}`` is the total count of term ``t`` in class ``c``, ``N_c`` is
the total count of all terms in class ``c``, ``V`` is the vocabulary size, and
``alpha`` is the smoothing parameter. We never compute the normalising term
(the evidence), since it is constant across classes for a given document.

The API deliberately mirrors scikit-learn (``fit`` / ``predict`` /
``predict_log_proba`` / ``predict_proba``) so it can be dropped into the same
evaluation harness and compared head-to-head with ``sklearn.MultinomialNB``.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


class MultinomialNaiveBayes:
    """Multinomial Naive Bayes with Laplace smoothing.

    Parameters
    ----------
    alpha:
        Additive (Laplace/Lidstone) smoothing parameter. ``alpha=1.0`` is
        classic Laplace smoothing; smaller values lean more on the data.
    fit_prior:
        If ``True`` (default) learn class priors from the training class
        frequencies. If ``False`` use a uniform prior over classes.
    """

    def __init__(self, alpha: float = 1.0, fit_prior: bool = True):
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = float(alpha)
        self.fit_prior = fit_prior

    # ------------------------------------------------------------------ fit
    def fit(self, X, y) -> "MultinomialNaiveBayes":
        """Estimate class log-priors and per-class feature log-likelihoods.

        ``X`` is an ``(n_samples, n_features)`` matrix of non-negative counts
        (dense ndarray or scipy sparse). ``y`` is an array of class labels.
        """
        X = self._as_float_2d(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have mismatched sample counts")

        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]
        n_features = X.shape[1]
        self.n_features_in_ = n_features

        # Sum term counts per class -> feature_count_[c, t] = N_{tc}.
        feature_count = np.zeros((n_classes, n_features), dtype=np.float64)
        class_count = np.zeros(n_classes, dtype=np.float64)
        for idx, cls in enumerate(self.classes_):
            mask = y == cls
            X_c = X[mask]
            # X_c.sum(axis=0) works for both dense and sparse; flatten to 1D.
            feature_count[idx, :] = np.asarray(X_c.sum(axis=0)).ravel()
            class_count[idx] = mask.sum()

        self.feature_count_ = feature_count
        self.class_count_ = class_count

        # Smoothed feature log-probabilities: log P(t | c).
        smoothed_fc = feature_count + self.alpha
        smoothed_class = smoothed_fc.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_class)

        # Class log-priors: log P(c).
        if self.fit_prior:
            self.class_log_prior_ = np.log(class_count) - np.log(class_count.sum())
        else:
            self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))

        return self

    # -------------------------------------------------------------- predict
    def _joint_log_likelihood(self, X) -> np.ndarray:
        """Return the unnormalised log P(c) + Σ x_t log P(t|c) per class."""
        X = self._as_float_2d(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )
        # (n_samples, n_features) @ (n_features, n_classes) -> (n_samples, n_classes)
        jll = X @ self.feature_log_prob_.T
        jll = np.asarray(jll)
        return jll + self.class_log_prior_

    def predict_log_proba(self, X) -> np.ndarray:
        """Return normalised class log-probabilities (log-sum-exp normalised)."""
        jll = self._joint_log_likelihood(X)
        # log-sum-exp over classes for the normaliser, done stably.
        log_norm = self._logsumexp(jll, axis=1, keepdims=True)
        return jll - log_norm

    def predict_proba(self, X) -> np.ndarray:
        """Return normalised class probabilities."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X) -> np.ndarray:
        """Return the most probable class label for each row of ``X``."""
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def score(self, X, y) -> float:
        """Mean accuracy on ``(X, y)``."""
        y = np.asarray(y)
        return float((self.predict(X) == y).mean())

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _as_float_2d(X):
        """Coerce input to a 2-D float matrix, keeping sparse matrices sparse."""
        if sparse.issparse(X):
            return X.astype(np.float64)
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X

    @staticmethod
    def _logsumexp(a: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
        a_max = np.max(a, axis=axis, keepdims=True)
        out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
        out = out + a_max
        if not keepdims:
            out = np.squeeze(out, axis=axis)
        return out
