# claims-synth

Synthetic healthcare claim generator + rules-based payer adjudicator for denial-risk modeling.

**Zero copyrighted code descriptions.** ICD-10 + HCPCS Level II only as shipped vocab. CPT treated as opaque user-supplied IDs.

## G1 — Claim Generator ✅

```
python -m claims_synth.generate --n 500 --seed 7
```

Produces deterministic, reproducible synthetic claims (837I + 837P) with:
- ICD-10-CM diagnoses (curated public-domain subset, 80+ codes)
- HCPCS Level II procedures (drugs, DME, supplies, transport, temp codes, modifiers)
- CPT support via `--cpt-codes` (opaque user-supplied IDs)
- Payer profiles (8 opaque payers, configurable weights)
- Encounter data (POS, admission type, discharge status, LOS)
- Authorization fields (auth number, status, referral)
- Full validation (structural, charge consistency, pointer integrity)

## G2 — Rules-Based Payer Adjudicator ✅

```python
from claims_synth.generate import ClaimGenerator
from claims_synth.adjudicate import Adjudicator, adjudicate_batch, summarize

# Generate claims
gen = ClaimGenerator(seed=42)
claims = gen.generate(n=100)

# Adjudicate
adj = Adjudicator(seed=42)
result = adj.adjudicate(claims[0])
print(result.to_json())

# Batch with summary
results = adjudicate_batch(claims, seed=42)
stats = summarize(results)
print(f"Payment rate: {stats['payment_rate']:.2%}")
print(f"Line denial rate: {stats['line_denial_rate']:.2%}")
```

Rules-based adjudication producing 835-like remittance advice:
- **8 payer profiles** with distinct denial rates, strictness, bundling, deductibles
- **Authorization checks** — denied/missing auth → CO-197/198 denial
- **Timely filing** — claims past deadline → PR-29 denial
- **Medical necessity** — diagnosis-procedure compatibility matrix (ICD-10 chapter × HCPCS group)
- **Place of service** — procedure not payable in setting → CO-B15 denial
- **Bundling** — related procedures bundled into primary → CO-97
- **Payment calculation** — deductible, payer share, patient responsibility
- **Standard reason codes** — CMS public-domain CARC/RARC codes (CO/PR/OA/PI)
- **Deterministic by seed** — same claim + same seed = same result
- **Per-payer summary statistics** — payment rate, denial rate by payer

## Roadmap

| Goal | Description | Status |
|------|-------------|--------|
| G1 | Claim object + generator | ✅ |
| G2 | Rules-based payer adjudicator → labeled 835s | ✅ |
| G3 | Heterogeneous stochastic payer profiles | ⬜ |
| G4 | Feature degradation (anti-leakage harness) | ⬜ |
| G5 | Denial-risk model, gated on held-out payer | ⬜ |
| G6 | Friction coefficients + negotiation parity report | ⬜ |
| SC | Fidelity scorecard vs CMS DE-SynPUF | ✅ |

## License

MIT. No AMA-copyrighted CPT content shipped.

## Fidelity Scorecard — graded against CMS DE-SynPUF ✅

How realistic is the synthetic output? claims-synth's claim shapes are modeled after
CMS-style claims data, and the scorecard makes that measurable: it grades generated claims
against **aggregate distributions computed from the CMS 2008–2010 Data Entrepreneurs'
Synthetic Public Use File (DE-SynPUF)** — 66,773 Medicare inpatient claim records and 116,352 beneficiaries (DE1.0,
Sample 1). DE-SynPUF is itself synthetic: CMS derived it from real 2008–2010 Medicare
fee-for-service claims and published it in the public domain, no data use agreement required,
precisely so software can be built against realistic claim shapes without touching restricted
data. No model, no LLM: every subscore is an objective distributional distance
(Kolmogorov–Smirnov for continuous fields, Jensen–Shannon for categorical).

```
python -m claims_synth.scorecard --realistic --claim-type institutional --n 3000 --seed 7
```

Writes `scorecard.json` and `scorecard.md`: a subscore per dimension (total charge, length of
stay, diagnoses per claim, lines per claim, primary-diagnosis chapter mix), an overall 0–100
fidelity number, the validity rate, and the full reference provenance. Run it on your own
configuration and judge the output for yourself — the scorecard is a measurement, not a
certification.

### The `--realistic` preset

`ClaimGenerator.realistic_inpatient()` (CLI: `--realistic`) **calibrates the generator directly
to the bundled reference aggregates**: total charge, length of stay, diagnosis count, line
count, and the primary-diagnosis chapter mix are each sampled from the DE-SynPUF reference
distribution itself (837I only), subject to structural validity — claims must carry ≥1 line and
≥1 diagnosis, and charges are non-negative. The preset is opt-in; the default generator is
unchanged.

```python
from claims_synth.generate import ClaimGenerator
gen = ClaimGenerator.realistic_inpatient(seed=7)
```

Two design notes:

- **The line-count dimension has a structural ceiling.** The reference counts ICD procedures,
  and ~43% of DE-SynPUF inpatient claims have zero — but claims-synth lines carry the charges
  (and real-world 837I requires a service line), so calibrated claims always carry at least one
  line; a perfect match on that dimension is impossible by design. The preset uses the
  JS-optimal reallocation of that mass over counts ≥1.
- **This is marginal calibration, not held-out generalization.** The preset samples from the
  same public aggregates the scorecard grades against; the scorecard measures how faithfully
  the full generative pipeline — claim assembly, validation, feature extraction — reproduces
  the reference marginals, with residual distance being sampling noise plus the structural
  ceiling above. Joint structure (e.g., charge × LOS correlation) is not yet calibrated or
  scored; that is the natural next dimension.

### Building the reference

The repo bundles a small (~20 KB) reference at
`src/claims_synth/scorecard/data/desynpuf_refstats.json` containing **aggregate distributional
summaries only** — per-dimension pmfs and a payment quantile grid; no claim rows, no
identifiers, no linkable records. To rebuild it from the raw DE-SynPUF yourself:

```
# download DE1.0 Sample 1 (public domain, no DUA required) from CMS, unzip, then:
python -m claims_synth.scorecard.build_reference \
    --inpatient   DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv \
    --beneficiary DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv \
    --out src/claims_synth/scorecard/data/desynpuf_refstats.json
```

Reference source:
[CMS 2008–2010 DE-SynPUF](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf)
— published by CMS as a public use file with no beneficiary PHI. claims-synth is not affiliated
with or endorsed by CMS.

### Comparability caveats (read before trusting a number)

- **DE-SynPUF is synthetic.** CMS notes it has "very limited inferential research value to draw
  conclusions about Medicare beneficiaries due to the synthetic processes used to create the
  file." Matching its marginals means matching CMS's public reference shapes — it is not a
  certification of equivalence to live Medicare data.
- **Amount** compares claims-synth *submitted charges* against DE-SynPUF *paid amounts* —
  related but not identical; treat the amount subscore as a shape check, not a dollar match.
- DE-SynPUF is **inpatient** Medicare fee-for-service; claims-synth's default stream is a
  **mix** of 837I/837P. Use `--claim-type institutional` to score 837I claims against the
  inpatient reference — an apples-to-apples comparison (same-day professional visits otherwise
  dilute the length-of-stay comparison).
- DE-SynPUF is **ICD-9** (2008–2010); claims-synth is **ICD-10-CM**. The `dx_chapter` dimension
  bridges them via a coarse common body-system chapter taxonomy, so it compares *mix*, not
  codes.
