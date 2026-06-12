# claims-synth

Synthetic healthcare claim generator + rules-based payer adjudicator for denial-risk modeling.

**Zero copyrighted code descriptions.** ICD-10 + HCPCS Level II only as shipped vocab. CPT treated as opaque user-supplied IDs.

## G1 — Claim Generator (current)

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

## Install

```
pip install -e ".[dev]"
```

## Test

```
pytest tests/ -v
```

## Roadmap

| Goal | Description | Status |
|------|-------------|--------|
| G1 | Claim object + generator | ✅ |
| G2 | Rules-based payer adjudicator → labeled 835s | ⬜ |
| G3 | Heterogeneous stochastic payer profiles | ⬜ |
| G4 | Feature degradation (anti-leakage harness) | ⬜ |
| G5 | Denial-risk model, gated on held-out payer | ⬜ |
| G6 | Friction coefficients + negotiation parity report | ⬜ |

## License

MIT. No AMA-copyrighted CPT content shipped.
