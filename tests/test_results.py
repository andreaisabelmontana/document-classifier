"""If a real training run has been done, results.json must clear a sane floor.

This test is skipped when results.json is absent (e.g. on a fresh checkout
before train.py has run), so the suite passes without requiring a 30-second
training run. After `python train.py`, it asserts both models beat a floor
well above the 5-class random baseline of 0.20.
"""

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parent.parent / "results.json"

# A 5-category problem has a 0.20 random baseline. Both real models score far
# above this; 0.60 is a conservative floor that still proves the pipeline works.
ACCURACY_FLOOR = 0.60


@pytest.mark.skipif(not RESULTS.exists(), reason="run train.py to produce results.json")
def test_results_clear_accuracy_floor():
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    models = data["models"]
    assert set(models) == {"tfidf_logreg", "scratch_multinomial_nb"}

    for name, m in models.items():
        assert m["accuracy"] > ACCURACY_FLOOR, f"{name} accuracy too low: {m['accuracy']}"
        assert m["macro_f1"] > ACCURACY_FLOOR, f"{name} macro-F1 too low: {m['macro_f1']}"


@pytest.mark.skipif(not RESULTS.exists(), reason="run train.py to produce results.json")
def test_results_metadata_present():
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert data["dataset"] == "20 Newsgroups"
    assert len(data["categories"]) == 5
    assert data["n_train"] > 0 and data["n_test"] > 0
