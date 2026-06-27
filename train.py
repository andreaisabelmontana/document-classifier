"""Train and evaluate both classifiers on the 20 Newsgroups slice.

Fits two models on the same train split and scores them on the held-out test
split:

  1. A TF-IDF + LogisticRegression scikit-learn pipeline.
  2. The from-scratch numpy Multinomial Naive Bayes (`docclf.MultinomialNaiveBayes`)
     on top of the same TF-IDF features.

Outputs (all written next to this script):
  * model.joblib        — the fitted sklearn pipeline (for predict.py)
  * results.json        — real accuracy / macro-F1 / per-class report for both
  * figures/confusion.png — confusion matrices for both models, side by side
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend; no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from docclf import MultinomialNaiveBayes, build_linear_pipeline, load_dataset, target_names

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
MODEL_PATH = HERE / "model.joblib"
RESULTS_PATH = HERE / "results.json"


def _metrics(name, y_true, y_pred, labels, names):
    """Bundle the headline metrics + per-class report for one model."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=names, digits=4
    )
    report_dict = classification_report(
        y_true, y_pred, labels=labels, target_names=names, output_dict=True
    )
    print(f"\n=== {name} ===")
    print(f"accuracy : {acc:.4f}")
    print(f"macro-F1 : {macro_f1:.4f}")
    print(report)
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "report": report_dict,
    }


def main() -> None:
    print("Loading 20 Newsgroups (train/test, metadata stripped)...")
    train = load_dataset("train")
    test = load_dataset("test")
    names = target_names()
    labels = list(range(len(names)))
    print(f"categories : {names}")
    print(f"train docs : {len(train.data)}")
    print(f"test docs  : {len(test.data)}")

    # --- Model 1: TF-IDF + LogisticRegression (sklearn pipeline) -----------
    print("\nFitting TF-IDF + LogisticRegression pipeline...")
    pipe: Pipeline = build_linear_pipeline()
    pipe.fit(train.data, train.target)
    lr_pred = pipe.predict(test.data)
    lr_metrics = _metrics(
        "TF-IDF + LogisticRegression", test.target, lr_pred, labels, names
    )

    # --- Model 2: from-scratch Multinomial NB on the same TF-IDF features --
    # Reuse the fitted vectoriser so both models see identical features.
    print("Fitting from-scratch Multinomial Naive Bayes...")
    vectorizer: TfidfVectorizer = pipe.named_steps["tfidf"]
    X_train = vectorizer.transform(train.data)
    X_test = vectorizer.transform(test.data)
    nb = MultinomialNaiveBayes(alpha=0.1)
    nb.fit(X_train, train.target)
    nb_pred = nb.predict(X_test)
    nb_metrics = _metrics(
        "From-scratch Multinomial Naive Bayes", test.target, nb_pred, labels, names
    )

    # --- Confusion matrices, side by side ----------------------------------
    FIGURES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, title, pred in (
        (axes[0], "TF-IDF + LogisticRegression", lr_pred),
        (axes[1], "From-scratch Multinomial NB", nb_pred),
    ):
        cm = confusion_matrix(test.target, pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
        disp.plot(ax=ax, cmap="Greens", colorbar=False, xticks_rotation=45)
        ax.set_title(title)
    fig.suptitle("Confusion matrices — 20 Newsgroups (5 categories)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved confusion matrices -> {FIGURES / 'confusion.png'}")

    # --- Persist the sklearn pipeline for predict.py -----------------------
    joblib.dump({"pipeline": pipe, "target_names": names}, MODEL_PATH)
    print(f"Saved trained pipeline   -> {MODEL_PATH}")

    # --- results.json ------------------------------------------------------
    results = {
        "dataset": "20 Newsgroups",
        "categories": names,
        "n_train": len(train.data),
        "n_test": len(test.data),
        "models": {
            "tfidf_logreg": lr_metrics,
            "scratch_multinomial_nb": nb_metrics,
        },
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved metrics            -> {RESULTS_PATH}")

    print("\nSummary")
    print(
        f"  LogReg : acc={lr_metrics['accuracy']:.4f}  macro-F1={lr_metrics['macro_f1']:.4f}"
    )
    print(
        f"  NB     : acc={nb_metrics['accuracy']:.4f}  macro-F1={nb_metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
