"""Machine-learning utility of synthetic claims: TRTR vs TSTR.

Answers "can a model trained on the synthetic data predict real outcomes as
well as one trained on real data?" — the standard train-on-synthetic,
test-on-real protocol:

- TRTR: predictor reads real training rows, is evaluated on a held-out real
  test split (the ceiling).
- TSTR: the same predictor type reads synthetic rows instead, evaluated on
  the same real test split.
- utility_ratio = TSTR AUC / TRTR AUC (1.0 means the synthetic data carries
  the predictive signal of the real data; ratios are clamped at 0 when AUC
  degenerates).

The scorer is predictor-agnostic: pass `predictor_factory` returning any
object with sklearn-style fit/predict_proba (or fit/predict for regression
scores). Zero-shot tabular foundation models fit naturally here because they
need no hyperparameter tuning on either side of the comparison — e.g. TabFM
(https://github.com/google-research/tabfm), for which `load_tabfm_factory()`
is provided; installing it is optional and everything else in this package
works without it. Requires a machine with roughly 20+ GB of free RAM or a
GPU to load.

Tables are plain dicts of column name -> list of values (the package's
native shape); the target column holds the outcome. Only numpy is required.
"""
import numpy as np


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


def _split_xy(table, target):
    cols = {k: list(v) for k, v in table.items() if k != target}
    y = np.asarray(table[target])
    return cols, y


def _positive_scores(est, cols, positive):
    """Probability of the positive class from an sklearn-style estimator."""
    X = _to_frame(cols)
    proba = est.predict_proba(X)
    classes = list(getattr(est, "classes_", []))
    idx = classes.index(positive) if positive in classes else -1
    return np.asarray(proba)[:, idx]


def _to_frame(cols):
    try:
        import pandas as pd
        return pd.DataFrame(cols)
    except ImportError:
        return cols


def score(real_train, real_test, synth_train, target,
          predictor_factory, positive=None):
    """Return TRTR / TSTR AUCs and their ratio for a binary outcome.

    real_train / real_test / synth_train: dict of column -> list, each
    containing the `target` column. `positive` names the positive class
    (default: the greater of the two observed classes).
    """
    for name, tbl in (("real_train", real_train), ("real_test", real_test),
                      ("synth_train", synth_train)):
        if target not in tbl:
            raise ValueError(f"{name} is missing target column {target!r}")

    _, y_test = _split_xy(real_test, target)
    classes = sorted(set(np.asarray(y_test).tolist()))
    if len(classes) != 2:
        raise ValueError(f"binary outcome required, saw classes {classes}")
    pos = positive if positive is not None else classes[1]
    y_bin = (np.asarray(y_test) == pos).astype(int)
    test_cols, _ = _split_xy(real_test, target)

    out = {}
    for arm, train_tbl in (("trtr", real_train), ("tstr", synth_train)):
        cols, y_tr = _split_xy(train_tbl, target)
        est = predictor_factory()
        est.fit(_to_frame(cols), np.asarray(y_tr))
        auc = rank_auc(y_bin, _positive_scores(est, test_cols, pos))
        out[arm] = {"auc": round(auc, 4) if auc is not None else None,
                    "n_train": len(y_tr)}

    trtr, tstr = out["trtr"]["auc"], out["tstr"]["auc"]
    if trtr and tstr and trtr > 0.5:
        ratio = max(0.0, (tstr - 0.5) / (trtr - 0.5))
    else:
        ratio = None
    out["utility_ratio"] = round(ratio, 4) if ratio is not None else None
    out["n_test"] = int(np.asarray(y_bin).size)
    out["positive_class"] = pos
    return out


def load_tabfm_factory():
    """Convenience: a predictor_factory backed by TabFM (optional dependency).

    Loads the classification model once and shares it across arms so both
    comparisons use identical weights.
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
