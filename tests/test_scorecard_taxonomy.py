"""Contract for claims_synth.scorecard.taxonomy — ICD-9 & ICD-10 -> common body-system chapter.

DE-SynPUF is ICD-9 (2008-2010); claims-synth is ICD-10-CM. A coarse common chapter taxonomy
bridges them so diagnosis-chapter MIX is comparable across the two coding systems.
"""
from claims_synth.scorecard.taxonomy import DX_CHAPTERS, icd9_chapter, icd10_chapter


def test_chapters_include_core():
    for c in ("Infectious", "Neoplasms", "Circulatory", "Respiratory", "Injury", "Other"):
        assert c in DX_CHAPTERS


def test_icd9_known_codes():
    assert icd9_chapter("0389") == "Infectious"            # 038 septicemia
    assert icd9_chapter("4019") == "Circulatory"           # 401 hypertension
    assert icd9_chapter("25000") == "Endocrine/Metabolic"  # 250 diabetes
    assert icd9_chapter("486") == "Respiratory"            # pneumonia
    assert icd9_chapter("820") == "Injury"                 # 800-999 fracture
    assert icd9_chapter("V270") == "Other"                 # V-code
    assert icd9_chapter("E8889") == "Other"                # E-code


def test_icd10_known_codes():
    assert icd10_chapter("I10") == "Circulatory"
    assert icd10_chapter("E119") == "Endocrine/Metabolic"
    assert icd10_chapter("J449") == "Respiratory"
    assert icd10_chapter("C509") == "Neoplasms"            # C = neoplasm
    assert icd10_chapter("D12") == "Neoplasms"             # D00-D49 = neoplasm
    assert icd10_chapter("D649") == "Blood"                # D50-D89 = blood
    assert icd10_chapter("F209") == "Mental"
    assert icd10_chapter("O80") == "Pregnancy"
    assert icd10_chapter("S72001A") == "Injury"
    assert icd10_chapter("Z0000") == "Other"


def test_unknown_maps_to_other():
    for bad in ("", "zzz", "123"):
        assert icd9_chapter(bad) == "Other"
        assert icd10_chapter(bad) == "Other"


def test_all_outputs_are_valid_chapters():
    for c in ("0389", "4019", "25000", "486", "820", "V270", "E8889"):
        assert icd9_chapter(c) in DX_CHAPTERS
    for c in ("I10", "E119", "J449", "C509", "D12", "D649", "F209", "O80", "S72001A", "Z0000"):
        assert icd10_chapter(c) in DX_CHAPTERS
