"""Contract for claims_synth.scorecard.ml_utility — TRTR/TSTR predictive parity.

Two layers:

- The scorer is predictor-agnostic, so the core contract runs on a
  deterministic stub with no model dependency.
- The DEFAULT predictor is LightGBM; those tests skip when it is absent, which
  is also the CLI's degradation path.

The optional TabFM integration only needs to satisfy the same factory interface.
"""
import numpy as np
import pytest

from claims_synth.scorecard.ml_utility import (
    brier, ece, holdout_split, isotonic_apply, isotonic_fit, lightgbm_available,
    load_lightgbm_factory, rank_auc, reliability_curve, score, score_repeated,
)

requires_lightgbm = pytest.mark.skipif(not lightgbm_available(),
                                       reason="lightgbm is not installed")


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


def _big_tables(n=900, seed=11):
    """Larger tables with a categorical column, for the real LightGBM path."""
    rng = np.random.default_rng(seed)

    def draw(rs):
        sig = rs.normal(size=n)
        chap = rs.choice(["Circulatory", "Endocrine", "Injury"], size=n)
        lift = 0.8 * (chap == "Circulatory")
        y = (sig + lift + rs.normal(scale=0.5, size=n) > 0).astype(int)
        return {"signal": sig.tolist(), "noise": rs.normal(size=n).tolist(),
                "chapter": chap.tolist(), "outcome": y.tolist()}

    real = draw(rng)
    good = draw(np.random.default_rng(seed + 1))
    bad = dict(good)
    bad["outcome"] = np.random.default_rng(seed + 2).permutation(good["outcome"]).tolist()
    return real, good, bad


# ── Metrics ────────────────────────────────────────────────────────────────

def test_rank_auc_perfect_and_random():
    assert rank_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert rank_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert rank_auc([0, 0], [0.1, 0.2]) is None


def test_brier_bounds():
    assert brier([1, 1, 0, 0], [1.0, 1.0, 0.0, 0.0]) == 0.0
    assert brier([1, 1, 0, 0], [0.0, 0.0, 1.0, 1.0]) == 1.0
    assert brier([1, 0], [0.5, 0.5]) == 0.25
    assert brier([], []) is None


def test_ece_zero_when_calibrated():
    """A bin whose mean prediction equals its observed rate contributes nothing."""
    y = [0, 1] * 50
    p = [0.5] * 100
    assert ece(y, p, n_bins=5) == pytest.approx(0.0, abs=1e-12)


def test_ece_catches_an_overconfident_model():
    # ~0.9 predicted everywhere (distinct values so the binning is unambiguous)
    # against a true rate of 0.10, spread one positive per bin.
    p = [0.9 + i * 1e-6 for i in range(100)]
    y = [1 if i % 10 == 0 else 0 for i in range(100)]
    assert ece(y, p, n_bins=10) == pytest.approx(0.8, abs=1e-3)


def test_ece_bins_are_equal_frequency():
    rng = np.random.default_rng(3)
    p = rng.random(100).tolist()
    y = (rng.random(100) < 0.5).astype(int).tolist()
    sizes = [n for _, _, n in reliability_curve(y, p, n_bins=10)]
    assert sizes == [10] * 10


def test_auc_is_blind_to_miscalibration_but_ece_is_not():
    """The whole argument for the calibration column, pinned.

    A strictly increasing rescale of the scores cannot change any pairwise
    ordering, so AUC is identical to the last bit — while the probabilities it
    reports become badly wrong, which ECE and Brier both see.
    """
    rng = np.random.default_rng(5)
    n = 600
    p = rng.random(n)
    y = (rng.random(n) < p).astype(int)          # perfectly calibrated by construction
    squashed = 0.5 + 0.02 * (p - 0.5)            # monotone: same ranking, ~constant 0.5

    assert rank_auc(y, squashed) == rank_auc(y, p)
    assert ece(y, squashed) > 5 * ece(y, p)
    assert brier(y, squashed) > brier(y, p)


def test_isotonic_recovers_a_squashed_probability():
    rng = np.random.default_rng(6)
    n = 4000
    p = rng.random(n)
    y = (rng.random(n) < p).astype(int)
    squashed = 0.5 + 0.02 * (p - 0.5)
    curve = isotonic_fit(squashed, y)
    fixed = isotonic_apply(curve, squashed)
    # Rank is essentially untouched — isotonic only pools adjacent scores, and
    # pooling a misordered region can nudge AUC either way by a hair.
    assert rank_auc(y, fixed) == pytest.approx(rank_auc(y, squashed), abs=0.01)
    # Calibration, meanwhile, is repaired outright.
    assert ece(y, fixed) < ece(y, squashed) / 5


def test_isotonic_output_is_monotone_nondecreasing():
    x, fitted = isotonic_fit([3.0, 1.0, 2.0, 4.0], [1, 0, 1, 0])
    assert list(x) == sorted(x)
    assert all(b >= a - 1e-12 for a, b in zip(fitted, fitted[1:]))


# ── Scoring contract (stub predictor) ──────────────────────────────────────

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


def test_utility_ratio_is_chance_corrected_not_a_raw_auc_ratio():
    """A coin-flip TSTR arm must score 0, not TSTR/TRTR (which would flatter it)."""
    real, _, bad = _tables()
    sc = score(_half(real, True), _half(real, False), _half(bad, True),
               target="outcome", predictor_factory=StubPredictor)
    trtr, tstr = sc["trtr"]["auc"], sc["tstr"]["auc"]
    assert sc["utility_ratio"] == pytest.approx(max(0.0, (tstr - 0.5) / (trtr - 0.5)), abs=5e-4)
    assert sc["utility_ratio"] < tstr / trtr        # strictly harsher than the raw ratio


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


# ── Calibration + variance band ────────────────────────────────────────────

def test_every_metric_is_reported_for_both_arms():
    real, good, _ = _tables()
    sc = score(_half(real, True), _half(real, False), _half(good, True),
               target="outcome", predictor_factory=StubPredictor)
    for arm in ("trtr", "tstr"):
        for metric in ("auc", "brier", "ece"):
            assert sc[arm][metric] is not None, f"{arm}.{metric} missing"
    assert sc["ece_bins"] == 10
    assert sc["ece_binning"] == "equal-frequency"


def test_variance_band_accompanies_every_metric():
    real, good, _ = _tables()
    sc = score(_half(real, True), _half(real, False), _half(good, True),
               target="outcome", predictor_factory=StubPredictor, seeds=(0, 1, 2))
    assert sc["n_repeats"] == 3
    for key in ("trtr", "tstr"):
        for metric in ("auc", "brier", "ece"):
            for suffix in ("_sd", "_min", "_max"):
                assert metric + suffix in sc[key]
            assert sc[key][metric + "_min"] <= sc[key][metric] <= sc[key][metric + "_max"]
    for suffix in ("_sd", "_min", "_max"):
        assert "utility_ratio" + suffix in sc


def test_single_seed_band_is_degenerate_not_missing():
    """One repeat still reports a band — sd 0, min == max == the point estimate."""
    real, good, _ = _tables()
    sc = score(_half(real, True), _half(real, False), _half(good, True),
               target="outcome", predictor_factory=StubPredictor)
    assert sc["n_repeats"] == 1
    assert sc["trtr"]["auc_sd"] == 0.0
    assert sc["trtr"]["auc_min"] == sc["trtr"]["auc_max"] == sc["trtr"]["auc"]


def test_score_repeated_redraws_the_data_each_seed():
    seen = []

    def build(seed):
        real, good, _ = _tables(seed=seed)
        seen.append(seed)
        return _half(real, True), _half(real, False), _half(good, True)

    sc = score_repeated(build, target="outcome", predictor_factory=StubPredictor,
                        seeds=range(4))
    assert seen == [0, 1, 2, 3]
    assert sc["n_repeats"] == 4
    # a fresh draw of the data per seed must actually move the metrics — that
    # is the variance a single-split point estimate hides
    assert sc["trtr"]["auc_sd"] > 0
    assert sc["trtr"]["auc_max"] > sc["trtr"]["auc_min"]


def test_holdout_split_is_stratified_and_disjoint():
    real, _, _ = _tables(n=200)
    train, test = holdout_split(real, "outcome", seed=0, test_frac=0.25)
    assert len(test["outcome"]) + len(train["outcome"]) == 200
    assert len(test["outcome"]) == pytest.approx(50, abs=2)
    base = sum(real["outcome"]) / 200
    assert sum(test["outcome"]) / len(test["outcome"]) == pytest.approx(base, abs=0.05)
    # rows are moved, never copied: no signal value appears in both halves
    assert not (set(train["signal"]) & set(test["signal"]))


def test_training_arms_are_size_matched():
    """Otherwise a generator scores better merely by emitting more rows."""
    real, good, _ = _tables(n=120)
    big = {k: v * 3 for k, v in good.items()}
    sc = score(_half(real, True), _half(real, False), big,
               target="outcome", predictor_factory=StubPredictor)
    assert sc["tstr"]["n_train"] == sc["trtr"]["n_train"]


# ── Default predictor (LightGBM) ───────────────────────────────────────────

@requires_lightgbm
def test_lightgbm_is_the_default_predictor():
    real, good, _ = _big_tables()
    train, test = holdout_split(real, "outcome", seed=0)
    sc = score(train, test, good, target="outcome")
    assert sc["predictor"] == "lightgbm"
    assert sc["trtr"]["auc"] > 0.75
    assert sc["utility_ratio"] > 0.7


@requires_lightgbm
def test_lightgbm_handles_string_categorical_columns():
    real, good, _ = _big_tables()
    train, test = holdout_split(real, "outcome", seed=0)
    assert isinstance(train["chapter"][0], str)
    sc = score(train, test, good, target="outcome")
    assert sc["tstr"]["auc"] is not None


@requires_lightgbm
def test_lightgbm_default_reports_calibration_for_both_arms():
    real, good, _ = _big_tables()
    train, test = holdout_split(real, "outcome", seed=0)
    sc = score(train, test, good, target="outcome")
    for arm in ("trtr", "tstr"):
        assert "calibrated" in sc[arm]
        for metric in ("auc", "brier", "ece"):
            assert sc[arm]["calibrated"][metric] is not None


@requires_lightgbm
def test_lightgbm_default_sees_a_label_destroying_generator():
    real, _, bad = _big_tables()
    train, test = holdout_split(real, "outcome", seed=0)
    sc = score(train, test, bad, target="outcome")
    assert sc["utility_ratio"] < 0.35


@requires_lightgbm
def test_unseen_category_at_predict_time_does_not_crash():
    real, good, _ = _big_tables()
    train, test = holdout_split(real, "outcome", seed=0)
    test = dict(test)
    test["chapter"] = ["Neoplasms"] * len(test["chapter"])   # never seen in training
    sc = score(train, test, good, target="outcome")
    assert sc["tstr"]["auc"] is not None


def test_lightgbm_factory_error_is_actionable_when_absent(monkeypatch):
    """The CLI degrades on this message rather than crashing the scorecard."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "lightgbm":
            raise ImportError("no lightgbm")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert lightgbm_available() is False
    with pytest.raises(ImportError, match="pip install lightgbm"):
        load_lightgbm_factory()
