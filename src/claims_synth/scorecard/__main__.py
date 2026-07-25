"""CLI: grade claims-synth output against the CMS DE-SynPUF public reference.

    python -m claims_synth.scorecard --n 2000 --seed 7

Generates synthetic claims, extracts comparable features, scores fidelity vs the bundled
DE-SynPUF reference, adds the ML-utility (TRTR/TSTR) column, and writes scorecard.json +
scorecard.md.
"""
import argparse
import csv
import os
from importlib import resources

from claims_synth.generate import ClaimGenerator
from claims_synth.scorecard import ml_utility
from claims_synth.scorecard.features import (
    DENIAL_TARGET, dataset_features, denial_table, filter_claims, validity_rate,
)
from claims_synth.scorecard.fidelity import score
from claims_synth.scorecard.report import to_json, to_markdown
from claims_synth.scorecard.reference import load_reference

PROXY_NOTE = ("proxy: ClaimGenerator.realistic_inpatient (reference-calibrated) "
              "— NOT real claims; pass --ml-real-csv for a true TSTR")


def default_reference_path():
    try:
        p = resources.files("claims_synth.scorecard").joinpath("data/desynpuf_refstats.json")
        if os.path.exists(str(p)):
            return str(p)
    except Exception:
        pass
    return None


def _load_real_csv(path):
    """Load a real-claims feature table: numeric where parseable, else string."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} has no rows")
    if DENIAL_TARGET not in rows[0]:
        raise SystemExit(f"{path} must have a {DENIAL_TARGET!r} column (0/1 outcome)")
    cols = {}
    for key in rows[0]:
        vals = [r[key] for r in rows]
        try:
            cols[key] = [float(v) for v in vals]
        except ValueError:
            cols[key] = vals
    cols[DENIAL_TARGET] = [int(float(v)) for v in cols[DENIAL_TARGET]]
    return cols


def ml_utility_block(args):
    """TRTR/TSTR + calibration for the candidate generator, or a skip reason.

    Degrades gracefully: any missing dependency or degenerate population
    returns {"skipped": <why>} rather than failing the whole scorecard.
    """
    if args.no_ml_utility:
        return {"skipped": "disabled with --no-ml-utility"}
    if not ml_utility.lightgbm_available():
        return {"skipped": "lightgbm is not installed "
                           "(pip install lightgbm) — ML-utility column omitted"}

    real_csv = _load_real_csv(args.ml_real_csv) if args.ml_real_csv else None
    if real_csv is not None and len(set(real_csv[DENIAL_TARGET])) != 2:
        return {"skipped": f"{args.ml_real_csv} has a single-class {DENIAL_TARGET!r} column"}

    def build(seed):
        if real_csv is not None:
            real = real_csv
        else:
            real = denial_table(
                filter_claims(
                    ClaimGenerator.realistic_inpatient(seed=100_000 + seed).generate(n=args.ml_n),
                    args.claim_type),
                adj_seed=args.seed)
        train, test = ml_utility.holdout_split(real, DENIAL_TARGET, seed=seed,
                                               test_frac=args.ml_test_frac)
        gen = (ClaimGenerator.realistic_inpatient(seed=200_000 + seed) if args.realistic
               else ClaimGenerator(seed=200_000 + seed))
        synth = denial_table(filter_claims(gen.generate(n=args.ml_n), args.claim_type),
                             adj_seed=args.seed)
        return train, test, synth

    try:
        train, test, synth = build(0)
    except Exception as exc:                       # noqa: BLE001 - degrade, don't crash
        return {"skipped": f"could not build ML tables: {exc}"}
    for name, tbl in (("held-out real", test), ("candidate synthetic", synth)):
        if len(tbl[DENIAL_TARGET]) < 100:
            return {"skipped": f"only {len(tbl[DENIAL_TARGET])} {name} rows after the "
                               f"'{args.claim_type}' filter; raise --ml-n"}
        if len(set(tbl[DENIAL_TARGET])) != 2:
            return {"skipped": f"{name} rows are single-class on {DENIAL_TARGET!r}"}

    try:
        out = ml_utility.score_repeated(build, target=DENIAL_TARGET,
                                        seeds=range(args.ml_repeats),
                                        n_bins=args.ml_bins)
    except ImportError as exc:
        return {"skipped": str(exc)}
    out["target"] = f"{DENIAL_TARGET} (>=1 line denied by the rules-based adjudicator)"
    out["real_population"] = (f"file: {args.ml_real_csv}" if args.ml_real_csv else PROXY_NOTE)
    out["candidate_generator"] = ("ClaimGenerator.realistic_inpatient" if args.realistic
                                  else "ClaimGenerator (default)")
    return out


def main():
    ap = argparse.ArgumentParser(prog="claims_synth.scorecard",
                                 description="Grade synthetic claims against the CMS DE-SynPUF public reference.")
    ap.add_argument("--n", type=int, default=1000, help="claims to generate for scoring")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--reference", default=None,
                    help="path to a refstats.json (default: bundled DE-SynPUF reference)")
    ap.add_argument("--claim-type", choices=["all", "institutional", "professional"], default="all",
                    help="filter generated claims before scoring; 'institutional' (837I) matches the "
                         "DE-SynPUF inpatient reference for an apples-to-apples amount/LOS comparison")
    ap.add_argument("--realistic", action="store_true",
                    help="use the reference-calibrated realistic-inpatient generator preset "
                         "(837I marginals sampled from the bundled DE-SynPUF aggregates)")
    ap.add_argument("--no-ml-utility", action="store_true",
                    help="skip the ML-utility (TRTR/TSTR) column; it costs a few seconds "
                         "because it fits a model per arm per repeat")
    ap.add_argument("--ml-n", type=int, default=1500,
                    help="claims per population for the ML-utility column")
    ap.add_argument("--ml-repeats", type=int, default=5,
                    help="split-and-fit repeats behind the ML-utility variance band")
    ap.add_argument("--ml-test-frac", type=float, default=0.3,
                    help="held-out fraction of the real population for the ML-utility column")
    ap.add_argument("--ml-bins", type=int, default=ml_utility.DEFAULT_ECE_BINS,
                    help="equal-frequency bin count for ECE")
    ap.add_argument("--ml-real-csv", default=None,
                    help="CSV of REAL claim features plus a 'denied' 0/1 column. Without it the "
                         "held-out population is the reference-calibrated preset — a proxy, not "
                         "real claims, and the resulting ratio is a transfer check between two "
                         "synthetic populations")
    ap.add_argument("--out-json", default="scorecard.json")
    ap.add_argument("--out-md", default="scorecard.md")
    args = ap.parse_args()

    ref_path = args.reference or default_reference_path()
    if not ref_path:
        raise SystemExit("no reference found; pass --reference path/to/refstats.json")
    reference = load_reference(ref_path)

    gen = ClaimGenerator.realistic_inpatient(seed=args.seed) if args.realistic else ClaimGenerator(seed=args.seed)
    claims = filter_claims(gen.generate(n=args.n), args.claim_type)
    if not claims:
        raise SystemExit(f"no '{args.claim_type}' claims in the {args.n} generated; try a larger --n")

    feats = dataset_features(claims)
    vr = validity_rate(claims)
    sc = score(feats, reference, validity_rate=vr)
    sc["scored_claim_type"] = args.claim_type
    sc["scored_n"] = len(claims)

    ml = ml_utility_block(args)
    sc["ml_utility"] = ml
    if "skipped" in ml:
        print(f"ml-utility: skipped — {ml['skipped']}")

    pop = f"\n_Scored {len(claims)} `{args.claim_type}` claims (of {args.n} generated, seed {args.seed})._\n"
    with open(args.out_json, "w") as f:
        f.write(to_json(sc))
    with open(args.out_md, "w") as f:
        f.write(to_markdown(sc) + pop)
    print(to_markdown(sc) + pop)
    print(f"wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
