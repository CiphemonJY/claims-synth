"""Contract for claims_synth.scorecard.stats — objective distributional distance (self-contained)."""
from claims_synth.scorecard.stats import ks, js, to_dist


def test_ks_identical_is_zero():
    assert ks([1, 2, 3, 4], [1, 2, 3, 4]) == 0.0


def test_ks_bounds_and_empty():
    assert 0.0 <= ks([1, 2, 3], [10, 20, 30]) <= 1.0
    assert ks([], [1, 2]) == 1.0
    assert ks([1, 2], []) == 1.0


def test_js_identical_is_zero():
    assert abs(js({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}, ["a", "b"])) < 1e-9


def test_js_positive_bounded():
    v = js({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}, ["a", "b"])
    assert 0.0 < v <= 1.0


def test_to_dist_normalizes():
    d = to_dist([1, 1, 2, 3])
    assert abs(sum(d.values()) - 1.0) < 1e-9
    assert d["1"] == 0.5
