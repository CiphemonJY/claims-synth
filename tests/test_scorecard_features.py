"""Contract for claims_synth.scorecard.features — extract comparable features from Claim objects.

Comparable-to-DE-SynPUF dimensions: dollar amount, length of stay, #diagnoses, #procedures,
primary-diagnosis chapter. Plus validity (fraction of claims with no validate() errors).
"""
import pytest
from datetime import date
from claims_synth.claim import Claim, ClaimType, Diagnosis, ClaimLine, Encounter, PlaceOfService
from claims_synth.scorecard.features import dataset_features, claim_los, validity_rate, filter_claims


def _claim(dx_codes, charges, from_d, to_d):
    dx = [Diagnosis(code=c) for c in dx_codes]
    lines = [ClaimLine(sequence=i + 1, procedure_code="G0001", diagnosis_pointers=[1], charge=ch)
             for i, ch in enumerate(charges)]
    enc = Encounter(from_date=from_d, to_date=to_d, place_of_service=PlaceOfService.INPATIENT)
    return Claim(claim_type=ClaimType.INSTITUTIONAL, diagnoses=dx, lines=lines, encounter=enc)


def test_claim_los():
    assert claim_los(_claim(["I10"], [100.0], date(2010, 1, 1), date(2010, 1, 6))) == 5
    assert claim_los(_claim(["I10"], [100.0], date(2010, 1, 1), None)) == 0   # same-day / no thru date


def test_dataset_features_shapes_and_values():
    claims = [
        _claim(["I10", "E119"], [100.0, 50.0], date(2010, 1, 1), date(2010, 1, 4)),
        _claim(["J449"], [200.0], date(2010, 2, 1), date(2010, 2, 2)),
    ]
    f = dataset_features(claims)
    assert f["amount"] == [150.0, 200.0]
    assert f["los"] == [3, 1]
    assert f["n_dx"] == [2, 1]
    assert f["n_proc"] == [2, 1]
    assert f["dx_chapter"] == ["Circulatory", "Respiratory"]   # primary dx chapter (ICD-10)


def test_validity_rate_matches_validate():
    claims = [_claim(["I10"], [100.0], date(2010, 1, 1), date(2010, 1, 3)) for _ in range(3)]
    expected = sum(1 for c in claims if c.validate() == []) / len(claims)
    assert validity_rate(claims) == expected
    assert validity_rate([]) == 0.0


def _typed(ct):
    return Claim(claim_type=ct, diagnoses=[Diagnosis(code="I10")],
                 lines=[ClaimLine(sequence=1, procedure_code="G0001", diagnosis_pointers=[1], charge=100.0)],
                 encounter=Encounter(from_date=date(2010, 1, 1), to_date=date(2010, 1, 3),
                                     place_of_service=PlaceOfService.INPATIENT))


def test_filter_claims_by_type():
    claims = [_typed(ClaimType.INSTITUTIONAL), _typed(ClaimType.PROFESSIONAL), _typed(ClaimType.INSTITUTIONAL)]
    assert len(filter_claims(claims, "all")) == 3
    inst = filter_claims(claims, "institutional")
    assert len(inst) == 2 and all(c.claim_type == ClaimType.INSTITUTIONAL for c in inst)
    prof = filter_claims(claims, "professional")
    assert len(prof) == 1 and prof[0].claim_type == ClaimType.PROFESSIONAL


def test_filter_claims_unknown_raises():
    with pytest.raises(ValueError):
        filter_claims([], "bogus")
