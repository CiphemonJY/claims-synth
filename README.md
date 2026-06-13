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

## License

MIT. No AMA-copyrighted CPT content shipped.
