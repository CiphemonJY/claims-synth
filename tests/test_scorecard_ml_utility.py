"""Contract for claims_synth.scorecard.ml_utility — TRTR/TSTR predictive parity.

Predictor-agnostic: the scorer takes any fit/predict_proba factory, so the
contract runs on a deterministic stub with no model dependency. The optional
TabFM integration only needs to satisfy the same factory interface.
"""
import numpy as np
import pytest

from claims_synth.scorecard.ml_utility import rank_auc, score


class StubPredictor:
    """Scores rows by the 'signal' column, scaled by how correlated the
    training outcome was with it — so informative training data yields
    discriminative scores and shuffled training data yields noise."""

    def fit(self, X, y):
        sig = np.asarray(X["signal"] if isinstance(X, dict) else X["signal"].to_numpy(),
                         dtype=float)
        y = np.asarray(y, dtype=float)
        self.weight_ = float(np.corrcoef(sig, y)[0, 1]) if len(set(y)) > 1 else 0.0
        if np.isnan(self.weight_):
            self.weight_ = 0.0
        self.classes_ = np.array(sorted(set(np.asarray(y).tolist())))
        return self

    def predict_proba(self, X):
        sig = np.asarray(X["signal"] if isinstance(X, dict) else X["signal"].to_numpy(),
                         dtype=float)
        p1 = 1 / (1 + np.exp(-self.weight_ * (sig - sig.mean())))
        return np.column_stack([1 - p1, p1])


def _tables(n=120, seed=7):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = (signal + rng.normal(scale=0.4, size=n) > 0).astype(int)
    real = {"signal": signal.tolist(), "noise": noise.tolist(), "outcome": y.tolist()}
    # synthetic that keeps the signal
    good = {"signal": signal.tolist(), "noise": rng.normal(size=n).tolist(),
            "outcome": y.tolist()}
    # synthetic that destroys it
    bad = {"signal": rng.permutation(signal).tolist(), "noise": noise.tolist(),
           "outcome": rng.permutation(y).tolist()}
    return real, good, bad


def _half(t, first):
    n = len(t["outcome"]) // 2
    sl = slice(0, n) if first else slice(n, None)
    return {k: v[sl] for k, v in t.items()}


def test_rank_auc_perfect_and_random():
    assert rank_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert rank_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert rank_auc([0, 0], [0.1, 0.2]) is None


def test_signal_preserving_synth_scores_near_ceiling():
    real, good, _ = _tables()
    sc = score(_half(real, True), _half(real, False), _half(good, True),
               target="outcome", predictor_factory=StubPredictor)
    assert sc["trtr"]["auc"] > 0.8
    assert sc["utility_ratio"] > 0.8


def test_signal_destroying_synth_scores_low():
    real, _, bad = _tables()
    sc = score(_half(real, True), _half(real, False), _half(bad, True),
               target="outcome", predictor_factory=StubPredictor)
    assert sc["trtr"]["auc"] > 0.8
    assert sc["utility_ratio"] < 0.4


def test_missing_target_raises():
    real, good, _ = _tables()
    with pytest.raises(ValueError):
        score({"signal": [1.0]}, _half(real, False), _half(good, True),
              target="outcome", predictor_factory=StubPredictor)


def test_non_binary_outcome_raises():
    real, good, _ = _tables()
    multi = dict(_half(real, False))
    multi["outcome"] = [0, 1, 2] * (len(multi["outcome"]) // 3)
    with pytest.raises(ValueError):
        score(_half(real, True), multi, _half(good, True),
              target="outcome", predictor_factory=StubPredictor)


def test_module_imports_without_tabfm():
    """The TabFM adapter is optional; importing the module must never require it."""
    import claims_synth.scorecard.ml_utility as m
    assert hasattr(m, "load_tabfm_factory")
