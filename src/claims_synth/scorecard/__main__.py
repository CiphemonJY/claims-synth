"""CLI: grade claims-synth output against the CMS DE-SynPUF public reference.

    python -m claims_synth.scorecard --n 2000 --seed 7

Generates synthetic claims, extracts comparable features, scores fidelity vs the bundled
DE-SynPUF reference, and writes scorecard.json + scorecard.md.
"""
import argparse
import os
from importlib import resources

from claims_synth.generate import ClaimGenerator
from claims_synth.scorecard.features import dataset_features, validity_rate, filter_claims
from claims_synth.scorecard.fidelity import score
from claims_synth.scorecard.report import to_json, to_markdown
from claims_synth.scorecard.reference import load_reference


def default_reference_path():
    try:
        p = resources.files("claims_synth.scorecard").joinpath("data/desynpuf_refstats.json")
        if os.path.exists(str(p)):
            return str(p)
    except Exception:
        pass
    return None


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

    pop = f"\n_Scored {len(claims)} `{args.claim_type}` claims (of {args.n} generated, seed {args.seed})._\n"
    with open(args.out_json, "w") as f:
        f.write(to_json(sc))
    with open(args.out_md, "w") as f:
        f.write(to_markdown(sc) + pop)
    print(to_markdown(sc) + pop)
    print(f"wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
