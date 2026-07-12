"""Contract for claims_synth.scorecard.fidelity — grade candidate features vs a reference.

Per-dimension objective distance (1 - KS for continuous, 1 - JS for categorical/discrete) ->
subscores -> overall 0-100 fidelity + letter grade. No model, no LLM.
"""
from claims_synth.scorecard.fidelity import score


def _ref():
    return {
        "dx_chapters": ["Circulatory", "Respiratory", "Other"],
        "amount": {"kind": "continuous", "sorted": [100.0, 150.0, 200.0, 250.0, 300.0], "n": 5},
        "los": {"kind": "discrete", "dist": {"1": 0.5, "2": 0.5}},
        "n_dx": {"kind": "discrete", "dist": {"1": 0.5, "2": 0.5}},
        "n_proc": {"kind": "discrete", "dist": {"1": 1.0}},
        "dx_chapter": {"kind": "categorical", "dist": {"Circulatory": 0.5, "Respiratory": 0.5, "Other": 0.0}},
        "provenance": {"source": "CMS DE-SynPUF", "n_claims": 5},
    }


def test_close_match_scores_high():
    feats = {"amount": [100.0, 150.0, 200.0, 250.0, 300.0], "los": [1, 2], "n_dx": [1, 2],
             "n_proc": [1, 1], "dx_chapter": ["Circulatory", "Respiratory"]}
    sc = score(feats, _ref())
    assert 0 <= sc["overall_fidelity"] <= 100
    assert sc["overall_fidelity"] > 80
    assert set(sc["dimensions"]) >= {"amount", "los", "n_dx", "n_proc", "dx_chapter"}


def test_poor_match_scores_low():
    feats = {"amount": [10000.0, 10000.0, 10000.0], "los": [50, 50, 50], "n_dx": [9, 9, 9],
             "n_proc": [9, 9, 9], "dx_chapter": ["Other", "Other", "Other"]}
    sc = score(feats, _ref())
    assert sc["overall_fidelity"] < 50


def test_validity_rate_passed_through():
    feats = {"amount": [150.0], "los": [1], "n_dx": [1], "n_proc": [1], "dx_chapter": ["Circulatory"]}
    sc = score(feats, _ref(), validity_rate=0.9)
    assert sc["validity_rate"] == 0.9


def test_grade_consistent_with_overall():
    feats = {"amount": [100.0, 150.0, 200.0, 250.0, 300.0], "los": [1, 2], "n_dx": [1, 2],
             "n_proc": [1, 1], "dx_chapter": ["Circulatory", "Respiratory"]}
    sc = score(feats, _ref())
    o = sc["overall_fidelity"]
    expected = "A" if o >= 90 else "B" if o >= 80 else "C" if o >= 70 else "D" if o >= 60 else "F"
    assert sc["grade"] == expected
