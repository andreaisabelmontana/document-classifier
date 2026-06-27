"""Classify a new piece of text with the trained pipeline.

Usage
-----
    python predict.py --file path/to/document.txt
    python predict.py --text "some text to classify"
    echo "some text" | python predict.py        # reads stdin

Prints the predicted category and the class probabilities. Run train.py first
to produce model.joblib.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.joblib"


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        sys.exit(
            f"Model not found at {path}. Run `python train.py` first to train it."
        )
    bundle = joblib.load(path)
    return bundle["pipeline"], bundle["target_names"]


def read_text(args) -> str:
    if args.text is not None:
        return args.text
    if args.file is not None:
        return Path(args.file).read_text(encoding="utf-8", errors="ignore")
    # Fall back to stdin.
    data = sys.stdin.read()
    if not data.strip():
        sys.exit("No input text. Pass --text, --file, or pipe text via stdin.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a document by its text.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--text", help="Text to classify (inline).")
    src.add_argument("--file", help="Path to a text file to classify.")
    parser.add_argument(
        "--top", type=int, default=3, help="How many ranked classes to show."
    )
    args = parser.parse_args()

    pipeline, names = load_model()
    text = read_text(args)

    pred = pipeline.predict([text])[0]
    print(f"\nPredicted category: {names[pred]}")

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba([text])[0]
        ranked = sorted(zip(names, proba), key=lambda kv: kv[1], reverse=True)
        print("\nClass probabilities:")
        for name, p in ranked[: args.top]:
            bar = "#" * int(round(p * 30))
            print(f"  {name:<24} {p:6.2%}  {bar}")


if __name__ == "__main__":
    main()
