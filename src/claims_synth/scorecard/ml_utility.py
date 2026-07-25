"""Machine-learning utility of synthetic claims: TRTR vs TSTR.

Answers "can a model trained on the synthetic data predict real outcomes as
well as one trained on real data?" — the standard train-on-synthetic,
test-on-real protocol:

- TRTR: predictor reads real training rows, is evaluated on a held-out real
  test split (the ceiling).
- TSTR: the same predictor type reads synthetic rows instead, evaluated on
  the same real test split.
- utility_ratio is the CHANCE-CORRECTED ratio
  ``max(0, (TSTR_AUC - 0.5) / (TRTR_AUC - 0.5))``: it measures how much of the
  real data's discriminative signal above coin-flip the synthetic data
  carries. 1.0 means parity with the real-trained model, 0.0 means the
  synthetic data carries no usable signal. (The uncorrected TSTR/TRTR ratio
  flatters a useless generator — 0.5/0.9 reads as 0.56 when the honest answer
  is 0.0 — so it is not what this module reports.)

Rank is not enough
------------------
AUC is invariant to every monotone transform of the score, so a model whose
probabilities are perfectly ordered but wildly mis-scaled scores a perfect
utility_ratio. The downstream product here is a denial-risk model whose output
is consumed as a PROBABILITY (thresholded to route claims for review, and
multiplied by dollars to rank worklists), not as a ranking — so this module
reports **Brier** and **ECE** alongside AUC for both arms, plus an optional
isotonic-calibrated variant.

That the two are near-orthogonal is measured, not assumed: on an internal
508k-row benchmark with identical features, isotonic regression moved AUC by
0.00005 (0.87429 -> 0.87424) while cutting ECE by 33% (0.00856 -> 0.00571);
on the same benchmark's MLP arm, calibration left AUC unchanged to seven
decimals (0.7912224) while ECE fell 3.2x (0.12973 -> 0.04094). A rank-only
scorecard cannot see any of that.

Every metric is reported as a band
----------------------------------
A single split and a single fit produce a number with no scale: 0.82 vs 0.74
is uninterpretable without knowing the spread. `score` takes `seeds=` and
`score_repeated` re-runs the whole split-and-fit, and both report
mean / sd / min / max for every metric.

The spread was MEASURED, not assumed — see `MEASURED_RESOLUTION` and
`resolution_note()` below. At the CLI's defaults a utility-ratio difference
under ~0.38 is indistinguishable from noise, so the column is a coarse
instrument: it separates "the synthetic data carries the signal" from "it
does not", and it cannot rank two decent generators against each other.

Predictors
----------
The default predictor is LightGBM (`load_lightgbm_factory`), which fits in
milliseconds and is available in CI. The scorer stays predictor-agnostic:
pass any `predictor_factory` returning an object with sklearn-style
fit/predict_proba. Zero-shot tabular foundation models fit naturally here
because they need no hyperparameter tuning on either side of the comparison
— e.g. TabFM (https://github.com/google-research/tabfm), for which
`load_tabfm_factory()` is provided as an optional second opinion; installing
it is optional and everything else in this package works without it (it needs
Python >= 3.11 and roughly 20+ GB of free RAM or a GPU to load, so it is not
the default).

Tables are plain dicts of column name -> list of values (the package's
native shape); the target column holds the outcome. Only numpy is required.
"""
import inspect

import numpy as np

#: LightGBM defaults. Tuned for the few-thousand-row tables this scorecard
#: produces. On a ~500k-row table the fleet-proven set is
#: num_leaves=96, learning_rate=0.03, min_child_samples=200,
#: feature_fraction=0.8, bagging_fraction=0.8, lambda_l1=1.0, lambda_l2=5.0.
DEFAULT_LGBM_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
}

#: Equal-frequency bin count for ECE unless overridden.
DEFAULT_ECE_BINS = 10

#: Null-channel measurement of this column's own noise, so callers know what a
#: difference has to beat before it means anything. Produced by running a
#: TRUE-ZERO comparison — the candidate generator IS the held-out population,
#: so the honest answer is exactly "no difference" — repeatedly with fresh
#: seeds, at 1500 institutional claims per population, and taking the spread of
#: successive independent differences. `mde80` is the smallest effect
#: detectable at 80% power against that spread.
#:
#: Two things worth knowing from it: the utility ratio is UPWARD-BIASED at
#: modest AUC (it centred on 1.03-1.06 where the truth is 1.00, because it
#: divides by a small, noisy AUC-0.5), and resolution buys back exactly as
#: 1/sqrt(repeats) — 0.851 at one repeat, 0.378 at five.
MEASURED_RESOLUTION = {
    "measured": "2026-07-25, ClaimGenerator.realistic_inpatient, "
                "n=1500 institutional claims per population, true effect = 0",
    "repeats_1": {"reps": 30, "utility_ratio_null_sd": 0.3038,
                  "utility_ratio_mde80": 0.851, "tstr_auc_mde80": 0.0836,
                  "tstr_ece_mde80": 0.0643},
    "repeats_5": {"reps": 20, "utility_ratio_null_sd": 0.1348,
                  "utility_ratio_mde80": 0.378, "tstr_auc_mde80": 0.0684,
                  "tstr_ece_mde80": 0.0332},
}


def resolution_note(n_repeats):
    """What a utility-ratio difference has to beat at `n_repeats` to be real."""
    base = MEASURED_RESOLUTION["repeats_5"]["utility_ratio_mde80"]
    mde = base * (5.0 / max(1, int(n_repeats))) ** 0.5
    return (f"measured null channel: at {n_repeats} repeat(s) a utility-ratio "
            f"difference below ~{mde:.2f} is indistinguishable from noise "
            f"(80% power); resolution improves as 1/sqrt(repeats)")


# ── Metrics ────────────────────────────────────────────────────────────────

def rank_auc(y_true, scores):
    """AUC via the Mann-Whitney U statistic (rank-based, ties averaged)."""
    y = np.asarray(y_true).astype(float)
    s = np.asarray(scores).astype(float)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=float)
    ranks[order] = np.arange(1, order.size + 1)
    combined = np.concatenate([pos, neg])
    sorted_vals = np.sort(combined)
    # average ranks for ties
    for v in np.unique(combined):
        mask = combined == v
        if mask.sum() > 1:
            lo = np.searchsorted(sorted_vals, v, side="left") + 1
            hi = np.searchsorted(sorted_vals, v, side="right")
            ranks[mask] = (lo + hi) / 2.0
    r_pos = ranks[: pos.size].sum()
    u = r_pos - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def brier(y_true, probs):
    """Mean squared error of the predicted probability (lower is better).

    Unlike AUC this is a PROPER score: it is minimised only by the true
    probability, so it penalises both bad ranking and bad scaling.
    """
    y = np.asarray(y_true).astype(float)
    p = np.asarray(probs).astype(float)
    if y.size == 0:
        return None
    return float(np.mean((p - y) ** 2))


def ece(y_true, probs, n_bins=DEFAULT_ECE_BINS):
    """Expected calibration error over EQUAL-FREQUENCY bins (lower is better).

    Equal-frequency (rather than equal-width) bins keep every bin's estimate
    equally noisy, which matters because predicted probabilities pile up at
    the ends of [0, 1]. Returns sum over bins of
    ``(n_bin / n) * |mean(prob) - mean(outcome)|``.
    """
    y = np.asarray(y_true).astype(float)
    p = np.asarray(probs).astype(float)
    n = y.size
    if n == 0:
        return None
    n_bins = max(1, min(int(n_bins), n))
    order = np.argsort(p, kind="mergesort")
    total = 0.0
    for chunk in np.array_split(order, n_bins):
        if chunk.size == 0:
            continue
        total += (chunk.size / n) * abs(float(p[chunk].mean()) - float(y[chunk].mean()))
    return float(total)


def reliability_curve(y_true, probs, n_bins=DEFAULT_ECE_BINS):
    """Equal-frequency reliability table: [(mean_prob, observed_rate, n), ...]."""
    y = np.asarray(y_true).astype(float)
    p = np.asarray(probs).astype(float)
    n = y.size
    if n == 0:
        return []
    n_bins = max(1, min(int(n_bins), n))
    order = np.argsort(p, kind="mergesort")
    out = []
    for chunk in np.array_split(order, n_bins):
        if chunk.size == 0:
            continue
        out.append((round(float(p[chunk].mean()), 4),
                    round(float(y[chunk].mean()), 4),
                    int(chunk.size)))
    return out


# ── Isotonic calibration ───────────────────────────────────────────────────

def isotonic_fit(scores, y_true):
    """Pool-adjacent-violators isotonic fit; returns an (x, y) lookup curve."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y_true, dtype=float)
    if s.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    order = np.argsort(s, kind="mergesort")
    x = s[order]
    vals = []
    wts = []
    for v in y[order]:
        vals.append(float(v))
        wts.append(1.0)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2 = vals.pop(), wts.pop()
            v1, w1 = vals.pop(), wts.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
    fitted = np.empty_like(x)
    i = 0
    for v, w in zip(vals, wts):
        k = int(round(w))
        fitted[i:i + k] = v
        i += k
    return x, fitted


def isotonic_apply(curve, scores):
    """Apply an isotonic curve by linear interpolation (np.interp over (x, y))."""
    x, y = curve
    return np.interp(np.asarray(scores, dtype=float), x, y)


# ── Table plumbing ─────────────────────────────────────────────────────────

def _split_xy(table, target):
    cols = {k: list(v) for k, v in table.items() if k != target}
    y = np.asarray(table[target])
    return cols, y


def _to_frame(cols):
    try:
        import pandas as pd
        return pd.DataFrame(cols)
    except ImportError:
        return cols


def _as_cols(X):
    """Normalise a DataFrame or dict-of-columns back to a dict of lists."""
    if hasattr(X, "columns"):
        return {c: list(X[c]) for c in X.columns}
    return {k: list(v) for k, v in X.items()}


def _subset(table, idx):
    return {k: [v[i] for i in idx] for k, v in table.items()}


def _positive_scores(est, cols, positive):
    """Probability of the positive class from an sklearn-style estimator."""
    X = _to_frame(cols)
    proba = est.predict_proba(X)
    classes = list(getattr(est, "classes_", []))
    idx = classes.index(positive) if positive in classes else -1
    return np.asarray(proba)[:, idx]


def _make(factory, seed):
    """Instantiate a predictor, passing `seed` only if the factory takes one."""
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        params = {}
    if "seed" in params or any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return factory(seed=seed)
    return factory()


# ── Predictors ─────────────────────────────────────────────────────────────

class _LightGBMPredictor:
    """LightGBM classifier that accepts this package's dict-of-columns tables.

    String columns are encoded as LightGBM categoricals against the level set
    seen at fit time; levels unseen at predict time map to -1, which LightGBM
    treats as an unknown category rather than an ordinal value.
    """

    def __init__(self, seed=0, **params):
        self.seed = seed
        self.params = dict(DEFAULT_LGBM_PARAMS)
        self.params.update(params)

    def _matrix(self, cols):
        n = len(next(iter(cols.values())))
        X = np.empty((n, len(self.columns_)), dtype=float)
        for j, name in enumerate(self.columns_):
            vals = cols.get(name, [])
            if name in self.levels_:
                lut = self.levels_[name]
                X[:, j] = [lut.get(str(v), -1) for v in vals]
            else:
                X[:, j] = [float(v) if v is not None else np.nan for v in vals]
        # LightGBM's sklearn wrapper always exposes feature_names_in_, so
        # predicting on a bare ndarray trips sklearn's name-mismatch warning.
        # Naming the columns identically at fit and predict avoids it.
        try:
            import pandas as pd
            return pd.DataFrame(X, columns=list(self.columns_))
        except ImportError:
            return X

    def fit(self, X, y):
        import lightgbm as lgb
        cols = _as_cols(X)
        self.columns_ = sorted(cols)
        self.levels_ = {}
        for name in self.columns_:
            vals = cols[name]
            if any(isinstance(v, str) for v in vals):
                self.levels_[name] = {lv: i for i, lv in
                                      enumerate(sorted({str(v) for v in vals}))}
        cat_idx = [j for j, n in enumerate(self.columns_) if n in self.levels_]
        y_arr = np.asarray(y)
        self.classes_ = np.array(sorted(set(y_arr.tolist())))
        y_enc = np.searchsorted(self.classes_, y_arr)
        model = lgb.LGBMClassifier(random_state=self.seed, verbose=-1, **self.params)
        model.fit(self._matrix(cols), y_enc,
                  categorical_feature=cat_idx or "auto")
        self.model_ = model
        return self

    def predict_proba(self, X):
        proba = self.model_.predict_proba(self._matrix(_as_cols(X)))
        if proba.shape[1] == 1:  # single-class training data
            return np.column_stack([1.0 - proba[:, 0], proba[:, 0]])
        return proba


def lightgbm_available():
    """True when the default predictor can actually be constructed."""
    try:
        import lightgbm  # noqa: F401
    except Exception:
        return False
    return True


def load_lightgbm_factory(**params):
    """The DEFAULT predictor_factory: gradient-boosted trees, ms-scale fits.

    Raises ImportError with an actionable message when LightGBM is absent, so
    callers can degrade gracefully instead of crashing.
    """
    try:
        import lightgbm  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "LightGBM is not installed; it backs the default ML-utility "
            "predictor. Install it with `pip install lightgbm`, or pass your "
            "own `predictor_factory` (any object with sklearn-style "
            "fit/predict_proba)."
        ) from exc

    def factory(seed=0):
        return _LightGBMPredictor(seed=seed, **params)

    return factory


def load_tabfm_factory():
    """Optional second opinion: a predictor_factory backed by TabFM.

    Loads the classification model once and shares it across arms so both
    comparisons use identical weights. Not the default — TabFM is CPU-slow
    (order of a second per row on JAX/CPU) and needs ~20 GB of RAM, so it is
    unusable in CI; `load_lightgbm_factory()` is the default instead.
    """
    try:
        from tabfm import TabFMClassifier, tabfm_v1_0_0_jax as loader
    except ImportError as exc:
        raise ImportError(
            "TabFM is not installed. It is an optional dependency: "
            "git clone https://github.com/google-research/tabfm && "
            "pip install -e './tabfm[jax]' (needs Python >= 3.11 and ~20 GB "
            "free RAM or a GPU to load)."
        ) from exc
    model = loader.load("classification")

    def factory():
        return TabFMClassifier(model=model)

    return factory


# ── Aggregation ────────────────────────────────────────────────────────────

def _agg(values):
    """mean / sd / min / max over the non-None values, or None if all None."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    a = np.asarray(vals, dtype=float)
    return {
        "mean": round(float(a.mean()), 4),
        "sd": round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 4),
        "min": round(float(a.min()), 4),
        "max": round(float(a.max()), 4),
        "n": int(a.size),
    }


def _flatten(prefix, band, into):
    """Write mean under `prefix` and the band under `prefix_sd/_min/_max`."""
    if band is None:
        into[prefix] = None
        return
    into[prefix] = band["mean"]
    into[prefix + "_sd"] = band["sd"]
    into[prefix + "_min"] = band["min"]
    into[prefix + "_max"] = band["max"]


# ── Scoring ────────────────────────────────────────────────────────────────

def _crossfit_isotonic(train_cols, y_tr, pos, factory, seed, n_folds=3):
    """Out-of-fold isotonic curve fitted on the arm's OWN training rows.

    Cross-fitting keeps the reported model identical to the uncalibrated one
    (the curve is a post-hoc map, not a retrain on less data). The TSTR arm's
    curve is necessarily fitted on SYNTHETIC rows — which is exactly the
    deployment situation the scorecard is testing: if the synthetic data is
    mis-scaled, its own calibration set cannot fix that.
    """
    n = len(y_tr)
    if n < 4 * n_folds:
        return None
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    oof = np.empty(n, dtype=float)
    y_bin = (np.asarray(y_tr) == pos).astype(int)
    for fold in np.array_split(order, n_folds):
        held = set(int(i) for i in fold)
        keep = [i for i in range(n) if i not in held]
        if not keep or len(set(y_bin[keep].tolist())) < 2:
            return None
        sub = {k: [v[i] for i in keep] for k, v in train_cols.items()}
        est = _make(factory, seed)
        est.fit(_to_frame(sub), np.asarray([y_tr[i] for i in keep]))
        held_cols = {k: [v[i] for i in fold] for k, v in train_cols.items()}
        oof[fold] = _positive_scores(est, held_cols, pos)
    return isotonic_fit(oof, y_bin)


def _downsample(table, target, n_keep, seed):
    """Stratified downsample to `n_keep` rows, preserving the class mix."""
    y = np.asarray(table[target])
    n = y.size
    if n <= n_keep:
        return table
    rng = np.random.default_rng(seed + 977)
    keep = []
    for cls in sorted(set(y.tolist())):
        idx = rng.permutation(np.where(y == cls)[0])
        take = int(round(len(idx) * n_keep / n))
        keep.extend(idx[:max(1, take)].tolist())
    return _subset(table, sorted(keep))


def _run_once(real_train, real_test, synth_train, target, factory, pos,
              seed, n_bins, calibrate, match_train_size=True):
    """One split-fixed pass: fit both arms at `seed`, return raw metrics."""
    test_cols, _ = _split_xy(real_test, target)
    y_bin = (np.asarray(real_test[target]) == pos).astype(int)
    if match_train_size:
        # Unequal training sizes confound the comparison: a generator would
        # score better simply by emitting more rows. Both arms see the same n.
        n_ref = len(real_train[target])
        synth_train = _downsample(synth_train, target, n_ref, seed)
        real_train = _downsample(real_train, target,
                                 len(synth_train[target]), seed)

    result = {}
    for arm, train_tbl in (("trtr", real_train), ("tstr", synth_train)):
        cols, y_tr = _split_xy(train_tbl, target)
        est = _make(factory, seed)
        est.fit(_to_frame(cols), np.asarray(y_tr))
        scores = _positive_scores(est, test_cols, pos)
        entry = {
            "auc": rank_auc(y_bin, scores),
            "brier": brier(y_bin, scores),
            "ece": ece(y_bin, scores, n_bins),
            "n_train": len(y_tr),
        }
        if calibrate:
            curve = _crossfit_isotonic(cols, list(y_tr), pos, factory, seed)
            if curve is not None:
                cal = isotonic_apply(curve, scores)
                entry["calibrated"] = {
                    "auc": rank_auc(y_bin, cal),
                    "brier": brier(y_bin, cal),
                    "ece": ece(y_bin, cal, n_bins),
                }
        result[arm] = entry

    trtr, tstr = result["trtr"]["auc"], result["tstr"]["auc"]
    if trtr and tstr and trtr > 0.5:
        result["utility_ratio"] = max(0.0, (tstr - 0.5) / (trtr - 0.5))
    else:
        result["utility_ratio"] = None
    return result


_METRICS = ("auc", "brier", "ece")


def _assemble(runs, pos, n_test, seeds, n_bins, predictor_name):
    out = {}
    for arm in ("trtr", "tstr"):
        entry = {}
        for m in _METRICS:
            _flatten(m, _agg([r[arm][m] for r in runs]), entry)
        entry["n_train"] = runs[0][arm]["n_train"]
        cal_runs = [r[arm]["calibrated"] for r in runs if "calibrated" in r[arm]]
        if cal_runs:
            cal = {}
            for m in _METRICS:
                _flatten(m, _agg([c[m] for c in cal_runs]), cal)
            entry["calibrated"] = cal
        out[arm] = entry
    _flatten("utility_ratio", _agg([r["utility_ratio"] for r in runs]), out)
    out["n_test"] = int(n_test)
    out["positive_class"] = pos
    out["n_repeats"] = len(runs)
    out["seeds"] = list(seeds)
    out["ece_bins"] = int(n_bins)
    out["ece_binning"] = "equal-frequency"
    out["predictor"] = predictor_name
    out["resolution"] = resolution_note(len(runs))
    return out


def _resolve_factory(predictor_factory):
    if predictor_factory is not None:
        return predictor_factory, getattr(predictor_factory, "__name__", "custom")
    return load_lightgbm_factory(), "lightgbm"


def score(real_train, real_test, synth_train, target,
          predictor_factory=None, positive=None, seeds=(0,),
          n_bins=DEFAULT_ECE_BINS, calibrate=True, match_train_size=True):
    """TRTR / TSTR rank AND calibration metrics for a binary outcome.

    real_train / real_test / synth_train: dict of column -> list, each
    containing the `target` column. `positive` names the positive class
    (default: the greater of the two observed classes).

    `predictor_factory` defaults to LightGBM (`load_lightgbm_factory()`).
    `seeds` re-fits both arms at each seed against the SAME split, so the
    reported band covers fit variance; use `score_repeated` for a band that
    also covers the split and the draw of the data itself.
    `match_train_size` downsamples the larger training table so both arms see
    the same row count.

    Every metric is returned as `name` (mean) plus `name_sd`, `name_min`,
    `name_max`.
    """
    for name, tbl in (("real_train", real_train), ("real_test", real_test),
                      ("synth_train", synth_train)):
        if target not in tbl:
            raise ValueError(f"{name} is missing target column {target!r}")

    classes = sorted(set(np.asarray(real_test[target]).tolist()))
    if len(classes) != 2:
        raise ValueError(f"binary outcome required, saw classes {classes}")
    pos = positive if positive is not None else classes[1]

    factory, pname = _resolve_factory(predictor_factory)
    seeds = tuple(seeds) if not isinstance(seeds, int) else (seeds,)
    runs = [_run_once(real_train, real_test, synth_train, target, factory, pos,
                      s, n_bins, calibrate, match_train_size) for s in seeds]
    return _assemble(runs, pos, len(real_test[target]), seeds, n_bins, pname)


def score_repeated(build_tables, target, predictor_factory=None, positive=None,
                   seeds=range(10), n_bins=DEFAULT_ECE_BINS, calibrate=True,
                   match_train_size=True):
    """Full-variance band: re-draw the data AND the split at every seed.

    `build_tables(seed)` returns `(real_train, real_test, synth_train)` for
    that seed. This is the number to quote as the scorecard's resolution — it
    covers everything `score` does plus sampling of both populations and the
    train/test partition, which in practice dominates.
    """
    factory, pname = _resolve_factory(predictor_factory)
    runs, pos, n_test = [], None, 0
    for s in seeds:
        real_train, real_test, synth_train = build_tables(s)
        classes = sorted(set(np.asarray(real_test[target]).tolist()))
        if len(classes) != 2:
            raise ValueError(f"binary outcome required, saw classes {classes}")
        p = positive if positive is not None else classes[1]
        pos, n_test = p, len(real_test[target])
        runs.append(_run_once(real_train, real_test, synth_train, target,
                              factory, p, s, n_bins, calibrate,
                              match_train_size))
    if not runs:
        raise ValueError("score_repeated needs at least one seed")
    return _assemble(runs, pos, n_test, list(seeds), n_bins, pname)


def holdout_split(table, target, seed=0, test_frac=0.3):
    """Stratified train/test split of a dict-of-columns table."""
    y = np.asarray(table[target])
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in sorted(set(y.tolist())):
        idx = rng.permutation(np.where(y == cls)[0])
        cut = int(round(len(idx) * test_frac))
        test_idx.extend(idx[:cut].tolist())
        train_idx.extend(idx[cut:].tolist())
    train_idx = rng.permutation(np.asarray(train_idx, dtype=int)).tolist()
    test_idx = rng.permutation(np.asarray(test_idx, dtype=int)).tolist()
    return _subset(table, train_idx), _subset(table, test_idx)
