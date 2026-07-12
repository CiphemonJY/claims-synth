"""Build the bundled DE-SynPUF reference statistics for the fidelity scorecard.

Reads CMS 2008-2010 DE-SynPUF (DE1.0, Sample 1) inpatient-claims and beneficiary-summary
CSVs and aggregates them into a small, publishable reference (aggregate distributions only — no
row-level data). The raw DE-SynPUF is public-domain synthetic data with no beneficiary PHI.

Usage:
    python -m claims_synth.scorecard.build_reference \
        --inpatient DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv \
        --beneficiary DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv \
        --out src/claims_synth/scorecard/data/desynpuf_refstats.json
"""
import argparse
import csv
import os

from claims_synth.scorecard.reference import build_reference, save_reference

SOURCE = "CMS 2008-2010 Data Entrepreneurs' Synthetic Public Use File (DE-SynPUF), DE1.0, Sample 1"
LANDING = ("https://www.cms.gov/data-research/statistics-trends-and-reports/"
           "medicare-claims-synthetic-public-use-files/"
           "cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf")
URLS = [
    "https://www.cms.gov/research-statistics-data-and-systems/downloadable-public-use-files/"
    "synpufs/downloads/de1_0_2008_to_2010_inpatient_claims_sample_1.zip",
    "https://www.cms.gov/research-statistics-data-and-systems/downloadable-public-use-files/"
    "synpufs/downloads/de1_0_2008_beneficiary_summary_file_sample_1.zip",
]


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inpatient", required=True)
    ap.add_argument("--beneficiary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--accessed", default="2026-07-11")
    ap.add_argument("--downsample", type=int, default=1000)
    args = ap.parse_args()

    inpatient = _read_csv(args.inpatient)
    beneficiary = _read_csv(args.beneficiary)

    provenance = {
        "source": SOURCE,
        "publisher": "Centers for Medicare & Medicaid Services (CMS)",
        "license": "Public domain (synthetic data; no beneficiary PHI)",
        "landing_page": LANDING,
        "download_urls": URLS,
        "files": [os.path.basename(args.inpatient), os.path.basename(args.beneficiary)],
        "n_claims": len(inpatient),
        "n_beneficiaries": len(beneficiary),
        "accessed": args.accessed,
        "notes": ("Reference distributions computed from CMS DE-SynPUF inpatient claim records "
                  "(length of stay, payment amount, diagnosis count, procedure count, "
                  "primary-diagnosis chapter) and beneficiary demographics (age, sex). "
                  "ICD-9-CM diagnosis codes are mapped to a coarse common body-system chapter "
                  "taxonomy so the mix is comparable against claims-synth's ICD-10-CM output. "
                  "DE-SynPUF is a synthetic public use file CMS derived from real 2008-2010 "
                  "Medicare fee-for-service inpatient claims (ICD-9; public domain, no DUA, no "
                  "beneficiary PHI); CMS notes it has limited inferential research value due to "
                  "the synthetic process. Caveat: the amount dimension compares claim payment vs "
                  "claims-synth submitted charge."),
    }

    ref = build_reference(inpatient, beneficiary, provenance, downsample=args.downsample)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_reference(ref, args.out)

    print(f"wrote {args.out}")
    print(f"  claims={ref['amount']['n']}  beneficiaries={provenance['n_beneficiaries']}")
    print(f"  amount mean=${ref['amount']['mean']:.0f} median=${ref['amount']['median']:.0f}")
    print(f"  los mean={ref['los']['mean']:.2f} days")
    print(f"  age mean={ref['age']['mean']:.1f}  female_frac={ref['sex']['female_frac']:.3f}")
    top = sorted(ref["dx_chapter"]["dist"].items(), key=lambda kv: kv[1], reverse=True)[:5]
    print("  top dx chapters: " + ", ".join(f"{k} {v:.2%}" for k, v in top))


if __name__ == "__main__":
    main()
