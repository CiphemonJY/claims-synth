"""Contract for claims_synth.scorecard.reference — aggregate parsed DE-SynPUF rows into refstats.

The CSV *reading* lives in build_reference.py; this module does the pure aggregation of already-
parsed row dicts into a compact, publishable reference (small enough to bundle, aggregate-only).
"""
from claims_synth.scorecard.reference import build_reference, save_reference, load_reference

INPATIENT = [
    {"CLM_PMT_AMT": "1000", "CLM_UTLZTN_DAY_CNT": "3", "CLM_ADMSN_DT": "20100101",
     "NCH_BENE_DSCHRG_DT": "20100104", "ICD9_DGNS_CD_1": "4019", "ICD9_DGNS_CD_2": "25000",
     "ICD9_PRCDR_CD_1": "3893", "HCPCS_CD_1": ""},
    {"CLM_PMT_AMT": "2000", "CLM_UTLZTN_DAY_CNT": "1", "CLM_ADMSN_DT": "20100201",
     "NCH_BENE_DSCHRG_DT": "20100202", "ICD9_DGNS_CD_1": "486", "ICD9_PRCDR_CD_1": "", "HCPCS_CD_1": ""},
]
BENEFICIARY = [
    {"BENE_BIRTH_DT": "19300101", "BENE_SEX_IDENT_CD": "1"},
    {"BENE_BIRTH_DT": "19400101", "BENE_SEX_IDENT_CD": "2"},
]


def test_build_reference_structure_and_values():
    ref = build_reference(INPATIENT, BENEFICIARY, provenance={"source": "test"}, downsample=10)
    assert ref["amount"]["kind"] == "continuous" and ref["amount"]["n"] == 2
    assert ref["amount"]["sorted"] == sorted(ref["amount"]["sorted"])   # sorted ascending
    assert ref["los"]["kind"] == "discrete"
    assert abs(sum(ref["los"]["dist"].values()) - 1.0) < 1e-9
    # claim 1 has 2 diagnoses, claim 2 has 1
    assert ref["n_dx"]["dist"]["2"] == 0.5 and ref["n_dx"]["dist"]["1"] == 0.5
    # claim 1 primary dx 4019 -> Circulatory ; claim 2 primary 486 -> Respiratory
    assert ref["dx_chapter"]["dist"]["Circulatory"] == 0.5
    assert ref["dx_chapter"]["dist"]["Respiratory"] == 0.5
    assert ref["sex"]["female_frac"] == 0.5                              # one male, one female
    assert "provenance" in ref and "dx_chapters" in ref


def test_save_load_roundtrip(tmp_path):
    ref = build_reference(INPATIENT, BENEFICIARY, provenance={"source": "t"})
    p = str(tmp_path / "ref.json")
    save_reference(ref, p)
    ref2 = load_reference(p)
    assert ref2["amount"]["n"] == ref["amount"]["n"]
    assert ref2["dx_chapter"]["dist"] == ref["dx_chapter"]["dist"]
