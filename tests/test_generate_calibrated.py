"""Contract for the calibrated realistic-inpatient preset (and default-path stability)."""
import hashlib

from claims_synth.claim import ClaimType
from claims_synth.generate import ClaimGenerator, load_inpatient_calibration

# Date-free structural fingerprint of the DEFAULT generator (seed 42, 200 claims),
# captured before the calibrated preset existed. The default path must never drift.
GOLDEN_DEFAULT_FINGERPRINT = "4a884ac1ed87ef8a8869aac358a8ed2fd7812cc3cfb2ea1062b1b11934920013"


def _fingerprint(claims) -> str:
    parts = []
    for c in claims:
        parts.append("|".join([
            c.claim_id, c.claim_type.value, str(len(c.diagnoses)),
            ",".join(d.code for d in c.diagnoses),
            ",".join(f"{l.procedure_code}:{l.charge}:{l.units}" for l in c.lines),
            c.payer.payer_id, f"{c.total_charge:.2f}",
        ]))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def test_default_generator_unchanged():
    claims = ClaimGenerator(seed=42).generate(200)
    assert _fingerprint(claims) == GOLDEN_DEFAULT_FINGERPRINT


def test_calibrated_deterministic():
    a = ClaimGenerator.realistic_inpatient(seed=7).generate(50)
    b = ClaimGenerator.realistic_inpatient(seed=7).generate(50)
    assert _fingerprint(a) == _fingerprint(b)


def test_calibrated_all_institutional_and_valid():
    claims = ClaimGenerator.realistic_inpatient(seed=11).generate(300)
    assert all(c.claim_type == ClaimType.INSTITUTIONAL for c in claims)
    assert all(c.validate() == [] for c in claims)


def test_calibration_respects_structural_floors():
    cal = load_inpatient_calibration()
    assert "0" not in cal["n_dx_dist"] and "0" not in cal["n_line_dist"]
    assert min(cal["amount_grid"]) >= 0.0
    claims = ClaimGenerator.realistic_inpatient(seed=23).generate(200)
    assert all(len(c.lines) >= 1 and len(c.diagnoses) >= 1 for c in claims)
    assert all(c.total_charge >= 0 for c in claims)


def test_charges_exact_to_the_cent():
    claims = ClaimGenerator.realistic_inpatient(seed=5).generate(200)
    for c in claims:
        computed = sum(l.charge * l.units for l in c.lines)
        assert abs(computed - c.total_charge) < 0.005
