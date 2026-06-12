"""Tests for claims-synth G1: claim object + generator."""

import json
import pytest
from datetime import date

from claims_synth.claim import (
    Claim, ClaimLine, ClaimType, Diagnosis, Encounter,
    Payer, Authorization, PlaceOfService,
)
from claims_synth.generate import ClaimGenerator


# ── Claim object tests ─────────────────────────────────────────────────────

class TestClaimValidation:
    """Claim.validate() should catch structural errors."""

    def test_valid_professional_claim(self):
        claim = Claim(
            claim_id="C0001",
            claim_type=ClaimType.PROFESSIONAL,
            patient_id="PT12345",
            diagnoses=[Diagnosis(code="I10")],
            lines=[ClaimLine(sequence=1, procedure_code="99213", charge=150.0)],
            payer=Payer(payer_id="PAYER_A"),
        )
        assert claim.is_valid()
        assert claim.validate() == []

    def test_valid_institutional_claim(self):
        claim = Claim(
            claim_id="C0002",
            claim_type=ClaimType.INSTITUTIONAL,
            patient_id="PT12345",
            diagnoses=[Diagnosis(code="I10", present_on_admission=True)],
            lines=[ClaimLine(sequence=1, procedure_code="99221", charge=500.0, revenue_code="0510")],
            encounter=Encounter(from_date=date.today(), place_of_service=PlaceOfService.INPATIENT),
            payer=Payer(payer_id="PAYER_C", plan_type="MEDICARE"),
        )
        assert claim.is_valid()

    def test_missing_lines(self):
        claim = Claim(
            claim_id="C0003",
            claim_type=ClaimType.PROFESSIONAL,
            diagnoses=[Diagnosis(code="I10")],
            lines=[],
        )
        errors = claim.validate()
        assert any("at least one claim line" in e for e in errors)

    def test_missing_diagnoses(self):
        claim = Claim(
            claim_id="C0004",
            claim_type=ClaimType.PROFESSIONAL,
            diagnoses=[],
            lines=[ClaimLine(sequence=1, procedure_code="99213", charge=150.0)],
        )
        errors = claim.validate()
        assert any("at least one diagnosis" in e for e in errors)

    def test_line_sequence_mismatch(self):
        claim = Claim(
            claim_id="C0005",
            claim_type=ClaimType.PROFESSIONAL,
            diagnoses=[Diagnosis(code="I10")],
            lines=[
                ClaimLine(sequence=1, procedure_code="99213", charge=150.0),
                ClaimLine(sequence=3, procedure_code="85025", charge=25.0),  # should be 2
            ],
        )
        errors = claim.validate()
        assert any("sequence mismatch" in e for e in errors)

    def test_diagnosis_pointer_out_of_range(self):
        claim = Claim(
            claim_id="C0006",
            claim_type=ClaimType.PROFESSIONAL,
            diagnoses=[Diagnosis(code="I10")],
            lines=[ClaimLine(sequence=1, procedure_code="99213", charge=150.0, diagnosis_pointers=[1, 5])],
        )
        errors = claim.validate()
        assert any("diagnosis pointer" in e and "out of range" in e for e in errors)

    def test_institutional_requires_revenue_code(self):
        claim = Claim(
            claim_id="C0007",
            claim_type=ClaimType.INSTITUTIONAL,
            diagnoses=[Diagnosis(code="I10")],
            lines=[ClaimLine(sequence=1, procedure_code="99221", charge=500.0)],  # no revenue_code
            encounter=Encounter(from_date=date.today(), place_of_service=PlaceOfService.INPATIENT),
        )
        errors = claim.validate()
        assert any("revenue_code" in e for e in errors)

    def test_institutional_requires_encounter(self):
        claim = Claim(
            claim_id="C0008",
            claim_type=ClaimType.INSTITUTIONAL,
            diagnoses=[Diagnosis(code="I10")],
            lines=[ClaimLine(sequence=1, procedure_code="99221", charge=500.0, revenue_code="0510")],
            encounter=None,
        )
        errors = claim.validate()
        assert any("encounter" in e for e in errors)

    def test_total_charge_computed(self):
        claim = Claim(
            claim_id="C0009",
            claim_type=ClaimType.PROFESSIONAL,
            diagnoses=[Diagnosis(code="I10")],
            lines=[
                ClaimLine(sequence=1, procedure_code="99213", charge=150.0, units=2),
                ClaimLine(sequence=2, procedure_code="85025", charge=25.0, units=1),
            ],
        )
        assert claim.total_charge == 325.0  # 150*2 + 25*1

    def test_total_charge_mismatch_detected(self):
        claim = Claim(
            claim_id="C0010",
            claim_type=ClaimType.PROFESSIONAL,
            diagnoses=[Diagnosis(code="I10")],
            lines=[ClaimLine(sequence=1, procedure_code="99213", charge=150.0)],
            total_charge=999.0,  # wrong
        )
        errors = claim.validate()
        assert any("total_charge mismatch" in e for e in errors)

    def test_to_dict_roundtrip(self):
        claim = Claim(
            claim_id="C0011",
            claim_type=ClaimType.PROFESSIONAL,
            patient_id="PT12345",
            diagnoses=[Diagnosis(code="I10"), Diagnosis(code="E11.9")],
            lines=[
                ClaimLine(sequence=1, procedure_code="99213", charge=150.0, modifiers=["25"]),
                ClaimLine(sequence=2, procedure_code="85025", charge=25.0),
            ],
            encounter=Encounter(from_date=date(2026, 6, 12), place_of_service=PlaceOfService.OFFICE),
            payer=Payer(payer_id="PAYER_A", plan_type="COMMERCIAL"),
            authorization=Authorization(auth_number="AUTH123456", auth_status="approved"),
            statement_from=date(2026, 6, 12),
            statement_to=date(2026, 6, 12),
        )
        d = claim.to_dict()
        restored = Claim.from_dict(d)
        assert restored.claim_id == claim.claim_id
        assert restored.claim_type == claim.claim_type
        assert len(restored.diagnoses) == 2
        assert restored.diagnoses[0].code == "I10"
        assert len(restored.lines) == 2
        assert restored.lines[0].modifiers == ["25"]
        assert restored.encounter.from_date == date(2026, 6, 12)
        assert restored.payer.payer_id == "PAYER_A"
        assert restored.authorization.auth_number == "AUTH123456"
        assert restored.is_valid()

    def test_to_json_produces_valid_json(self):
        claim = Claim(
            claim_id="C0012",
            claim_type=ClaimType.PROFESSIONAL,
            diagnoses=[Diagnosis(code="I10")],
            lines=[ClaimLine(sequence=1, procedure_code="99213", charge=150.0)],
        )
        json_str = claim.to_json()
        parsed = json.loads(json_str)
        assert parsed["claim_id"] == "C0012"
        assert parsed["claim_type"] == "837P"


# ── Generator tests ────────────────────────────────────────────────────────

class TestGenerator:
    """ClaimGenerator produces valid, reproducible claims."""

    def test_generates_valid_claims(self):
        gen = ClaimGenerator(seed=42)
        claims = gen.generate(n=50)
        assert len(claims) == 50
        for claim in claims:
            assert claim.is_valid(), f"Claim {claim.claim_id} invalid: {claim.validate()}"

    def test_seed_reproducibility(self):
        gen1 = ClaimGenerator(seed=42)
        claims1 = gen1.generate(n=20)

        gen2 = ClaimGenerator(seed=42)
        claims2 = gen2.generate(n=20)

        for c1, c2 in zip(claims1, claims2):
            assert c1.claim_id == c2.claim_id
            assert c1.claim_type == c2.claim_type
            assert c1.patient_id == c2.patient_id
            assert len(c1.diagnoses) == len(c2.diagnoses)
            assert [d.code for d in c1.diagnoses] == [d.code for d in c2.diagnoses]
            assert len(c1.lines) == len(c2.lines)
            assert [l.procedure_code for l in c1.lines] == [l.procedure_code for l in c2.lines]
            assert c1.total_charge == c2.total_charge

    def test_different_seeds_produce_different_claims(self):
        gen1 = ClaimGenerator(seed=42)
        claims1 = gen1.generate(n=10)

        gen2 = ClaimGenerator(seed=99)
        claims2 = gen2.generate(n=10)

        # At least some claims should differ
        ids1 = {c.claim_id for c in claims1}
        ids2 = {c.claim_id for c in claims2}
        # IDs are sequential so they'll overlap, but content should differ
        codes1 = [c.lines[0].procedure_code for c in claims1]
        codes2 = [c.lines[0].procedure_code for c in claims2]
        assert codes1 != codes2, "Different seeds should produce different procedure codes"

    def test_mixed_claim_types(self):
        gen = ClaimGenerator(seed=42)
        claims = gen.generate(n=100)
        types = [c.claim_type for c in claims]
        inst = sum(1 for t in types if t == ClaimType.INSTITUTIONAL)
        prof = sum(1 for t in types if t == ClaimType.PROFESSIONAL)
        assert inst > 0, "Should generate some institutional claims"
        assert prof > 0, "Should generate some professional claims"
        assert prof > inst, "Professional should be majority (~70%)"

    def test_institutional_claims_have_encounter_and_revenue(self):
        gen = ClaimGenerator(seed=42)
        claims = gen.generate(n=100)
        inst_claims = [c for c in claims if c.claim_type == ClaimType.INSTITUTIONAL]
        for claim in inst_claims:
            assert claim.encounter is not None, f"Institutional claim {claim.claim_id} missing encounter"
            for line in claim.lines:
                assert line.revenue_code is not None, f"Institutional line missing revenue_code"

    def test_diagnoses_have_valid_codes(self):
        gen = ClaimGenerator(seed=42)
        claims = gen.generate(n=50)
        from claims_synth.vocab.icd10 import all_codes as icd10_all
        valid_codes = set(icd10_all())
        for claim in claims:
            for dx in claim.diagnoses:
                assert dx.code in valid_codes, f"Unknown ICD-10 code: {dx.code}"

    def test_procedures_have_valid_codes(self):
        gen = ClaimGenerator(seed=42)
        claims = gen.generate(n=50)
        from claims_synth.vocab.hcpcs import all_codes as hcpcs_all
        from claims_synth.vocab.cpt import all_codes as cpt_all
        valid_codes = set(hcpcs_all()) | set(cpt_all())
        for claim in claims:
            for line in claim.lines:
                assert line.procedure_code in valid_codes, f"Unknown code: {line.procedure_code}"

    def test_cpt_opaque_ids(self):
        """CPT codes can be mixed in as user-supplied opaque IDs."""
        gen = ClaimGenerator(seed=42, cpt_codes=["99213", "27130", "85025"], cpt_weight=0.5)
        claims = gen.generate(n=100)
        cpt_count = sum(1 for c in claims for l in c.lines if l.procedure_code_type == "CPT")
        hcpcs_count = sum(1 for c in claims for l in c.lines if l.procedure_code_type == "HCPCS")
        assert cpt_count > 0, "Should generate some CPT lines"
        assert hcpcs_count > 0, "Should generate some HCPCS lines"

    def test_cpt_only_mode(self):
        """cpt_weight=1.0 should use only CPT codes."""
        gen = ClaimGenerator(seed=42, cpt_codes=["99213", "99214", "99203"], cpt_weight=1.0)
        claims = gen.generate(n=20)
        for claim in claims:
            for line in claim.lines:
                assert line.procedure_code_type == "CPT", f"Expected CPT, got {line.procedure_code_type}"

    def test_payer_ids_from_pool(self):
        gen = ClaimGenerator(seed=42)
        claims = gen.generate(n=50)
        valid_payers = {"PAYER_A", "PAYER_B", "PAYER_C", "PAYER_D", "PAYER_E", "PAYER_F", "PAYER_G", "PAYER_H"}
        for claim in claims:
            assert claim.payer is not None
            assert claim.payer.payer_id in valid_payers

    def test_large_generation(self):
        """Stress test: 500 claims should all be valid."""
        gen = ClaimGenerator(seed=7)
        claims = gen.generate(n=500)
        assert len(claims) == 500
        invalid = [c for c in claims if not c.is_valid()]
        assert len(invalid) == 0, f"{len(invalid)} invalid claims out of 500"


# ── Vocab tests ────────────────────────────────────────────────────────────

class TestVocab:
    def test_icd10_lookup(self):
        from claims_synth.vocab.icd10 import lookup
        assert lookup("I10") is not None
        assert lookup("I10").category == "Essential hypertension"
        assert lookup("ZZZ999") is None

    def test_hcpcs_lookup(self):
        from claims_synth.vocab.hcpcs import lookup
        assert lookup("J0135") is not None
        assert lookup("J0135").is_drug
        assert lookup("E0601") is not None
        assert lookup("E0601").is_dme
        assert lookup("99999") is None

    def test_no_cpt_codes_in_hcpcs(self):
        """Verify no CPT (5-digit numeric) codes leaked into HCPCS vocab."""
        from claims_synth.vocab.hcpcs import HCPCS_CODES
        # CPT codes are 5-digit numeric (10000-99999). HCPCS Level II are letter-prefixed
        # or short modifier codes (2 chars). Flag any 5-digit all-numeric code.
        for c in HCPCS_CODES:
            is_5digit = c.code.isdigit() and len(c.code) == 5
            assert not is_5digit, \
                f"Potential CPT code in HCPCS vocab: {c.code}"

    def test_no_copyrighted_descriptions(self):
        """Quick sanity: vocab files shouldn't contain AMA copyright markers."""
        import claims_synth.vocab.icd10 as icd10
        import claims_synth.vocab.hcpcs as hcpcs
        import inspect
        icd10_src = inspect.getsource(icd10)
        hcpcs_src = inspect.getsource(hcpcs)
        for marker in ["©", "Copyright AMA", "CPT®", "Current Procedural Terminology"]:
            assert marker not in icd10_src, f"Copyright marker '{marker}' in icd10.py"
            assert marker not in hcpcs_src, f"Copyright marker '{marker}' in hcpcs.py"
