"""
ICD-10-CM diagnosis codes — public domain subset.

ICD-10 codes are published by WHO and CMS without copyright restriction.
We ship a curated subset of common codes organized by chapter.
No copyrighted descriptions — only code + short public-domain category label.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ICD10Code:
    code: str
    category: str          # public-domain short label (e.g. "Hypertension")
    chapter: str            # ICD-10 chapter (e.g. "Diseases of the circulatory system")
    is_chronic: bool = False
    is_cc: bool = False     # complication/comorbidity (MS-DRG CC)
    is_mcc: bool = False    # major CC


# ── Curated ICD-10-CM subset ──────────────────────────────────────────────
# Organized by chapter. Frequencies approximate CMS Medicare claims data
# (public domain aggregate statistics). CPT descriptions NOT included.

ICD10_CODES: list[ICD10Code] = [
    # Chapter 1: Infectious and parasitic diseases (A00-B99)
    ICD10Code("A41.9", "Sepsis, unspecified", "Infectious", is_mcc=True),
    ICD10Code("B20",   "HIV disease", "Infectious", is_chronic=True, is_cc=True),
    ICD10Code("B95.61", "Methicillin-susceptible Staph aureus", "Infectious", is_cc=True),
    ICD10Code("B95.62", "MRSA infection", "Infectious", is_mcc=True),
    ICD10Code("J18.9", "Pneumonia, unspecified", "Respiratory", is_cc=True),
    ICD10Code("U07.1", "COVID-19", "Infectious", is_cc=True),

    # Chapter 2: Neoplasms (C00-D49)
    ICD10Code("C50.919", "Breast cancer, unspecified", "Neoplasms", is_chronic=True, is_cc=True),
    ICD10Code("C34.90", "Lung cancer, unspecified", "Neoplasms", is_chronic=True, is_mcc=True),
    ICD10Code("C61",    "Prostate cancer", "Neoplasms", is_chronic=True, is_cc=True),
    ICD10Code("C18.9",  "Colon cancer, unspecified", "Neoplasms", is_chronic=True, is_mcc=True),
    ICD10Code("D50.9",  "Iron deficiency anemia, unspecified", "Blood", is_cc=True),
    ICD10Code("D64.9",  "Anemia, unspecified", "Blood"),

    # Chapter 4: Endocrine, nutritional, metabolic (E00-E89)
    ICD10Code("E11.9",  "Type 2 diabetes, uncomplicated", "Endocrine", is_chronic=True),
    ICD10Code("E11.65", "Type 2 diabetes with hyperglycemia", "Endocrine", is_chronic=True, is_cc=True),
    ICD10Code("E11.22", "Type 2 diabetes with CKD", "Endocrine", is_chronic=True, is_mcc=True),
    ICD10Code("E10.9",  "Type 1 diabetes, uncomplicated", "Endocrine", is_chronic=True),
    ICD10Code("E78.5",  "Hyperlipidemia, unspecified", "Endocrine", is_chronic=True),
    ICD10Code("E78.0",  "Pure hypercholesterolemia", "Endocrine", is_chronic=True),
    ICD10Code("E66.9",  "Obesity, unspecified", "Endocrine", is_chronic=True, is_cc=True),
    ICD10Code("E03.9",  "Hypothyroidism, unspecified", "Endocrine", is_chronic=True),

    # Chapter 5: Mental and behavioral (F01-F99)
    ICD10Code("F32.9",  "Major depressive disorder, unspecified", "Mental", is_chronic=True),
    ICD10Code("F41.9",  "Anxiety disorder, unspecified", "Mental", is_chronic=True),
    ICD10Code("F33.9",  "Recurrent depressive disorder, unspecified", "Mental", is_chronic=True),
    ICD10Code("F20.9",  "Schizophrenia, unspecified", "Mental", is_chronic=True, is_cc=True),

    # Chapter 6: Nervous system (G00-G99)
    ICD10Code("G40.909", "Epilepsy, unspecified", "Nervous", is_chronic=True, is_cc=True),
    ICD10Code("G35",    "Multiple sclerosis", "Nervous", is_chronic=True, is_cc=True),
    ICD10Code("G89.4",  "Chronic pain syndrome", "Nervous", is_chronic=True),

    # Chapter 7: Eye (H00-H59)
    ICD10Code("H25.9",  "Age-related cataract, unspecified", "Eye"),
    ICD10Code("H40.9",  "Glaucoma, unspecified", "Eye", is_chronic=True),

    # Chapter 9: Circulatory (I00-I99)
    ICD10Code("I10",    "Essential hypertension", "Circulatory", is_chronic=True),
    ICD10Code("I25.10", "CAD without angina", "Circulatory", is_chronic=True, is_cc=True),
    ICD10Code("I25.110","CAD with unstable angina", "Circulatory", is_chronic=True, is_mcc=True),
    ICD10Code("I50.9",  "Heart failure, unspecified", "Circulatory", is_chronic=True, is_mcc=True),
    ICD10Code("I48.91", "Atrial fibrillation, unspecified", "Circulatory", is_chronic=True, is_cc=True),
    ICD10Code("I63.9",  "Cerebral infarction, unspecified", "Circulatory", is_mcc=True),
    ICD10Code("I21.4",  "Non-ST elevation MI", "Circulatory", is_mcc=True),
    ICD10Code("I21.9",  "Acute MI, unspecified", "Circulatory", is_mcc=True),
    ICD10Code("I11.0",  "Hypertensive heart disease with HF", "Circulatory", is_chronic=True, is_mcc=True),
    ICD10Code("I70.219", "Atherosclerosis of extremities with claudication", "Circulatory", is_chronic=True, is_cc=True),

    # Chapter 10: Respiratory (J00-J99)
    ICD10Code("J44.9",  "COPD, unspecified", "Respiratory", is_chronic=True, is_cc=True),
    ICD10Code("J45.909", "Asthma, unspecified", "Respiratory", is_chronic=True),
    ICD10Code("J96.01", "Acute respiratory failure with hypoxia", "Respiratory", is_mcc=True),
    ICD10Code("J96.91", "Respiratory failure, unspecified", "Respiratory", is_mcc=True),

    # Chapter 11: Digestive (K00-K95)
    ICD10Code("K21.9",  "GERD without esophagitis", "Digestive", is_chronic=True),
    ICD10Code("K80.20", "Gallstones without obstruction", "Digestive"),
    ICD10Code("K85.9",  "Acute pancreatitis, unspecified", "Digestive", is_cc=True),
    ICD10Code("K57.30", "Diverticulosis without complication", "Digestive", is_chronic=True),
    ICD10Code("K57.32", "Diverticulitis without perforation", "Digestive", is_cc=True),
    ICD10Code("K76.0",  "Fatty liver disease", "Digestive", is_chronic=True),

    # Chapter 12: Skin (L00-L99)
    ICD10Code("L40.9",  "Psoriasis, unspecified", "Skin", is_chronic=True),
    ICD10Code("L20.9",  "Atopic dermatitis, unspecified", "Skin", is_chronic=True),
    ICD10Code("L97.909", "Non-pressure chronic ulcer, unspecified site", "Skin", is_chronic=True, is_cc=True),

    # Chapter 13: Musculoskeletal (M00-M99)
    ICD10Code("M17.9",  "Osteoarthritis of knee, unspecified", "Musculoskeletal", is_chronic=True),
    ICD10Code("M16.9",  "Osteoarthritis of hip, unspecified", "Musculoskeletal", is_chronic=True),
    ICD10Code("M54.5",  "Low back pain", "Musculoskeletal", is_chronic=True),
    ICD10Code("M54.16", "Radiculopathy, lumbar", "Musculoskeletal", is_chronic=True),
    ICD10Code("M51.26", "Other intervertebral disc displacement, lumbar", "Musculoskeletal", is_chronic=True),
    ICD10Code("M81.0",  "Age-related osteoporosis", "Musculoskeletal", is_chronic=True),
    ICD10Code("M79.7",  "Fibromyalgia", "Musculoskeletal", is_chronic=True),
    ICD10Code("M25.561", "Pain in right knee", "Musculoskeletal"),

    # Chapter 14: Genitourinary (N00-N99)
    ICD10Code("N18.3",  "CKD stage 3", "Genitourinary", is_chronic=True, is_cc=True),
    ICD10Code("N18.6",  "ESRD", "Genitourinary", is_chronic=True, is_mcc=True),
    ICD10Code("N39.0",  "UTI, site not specified", "Genitourinary"),
    ICD10Code("N40.1",  "BPH with LUTS", "Genitourinary", is_chronic=True),
    ICD10Code("N20.0",  "Calculus of kidney", "Genitourinary"),

    # Chapter 15: Pregnancy (O00-O9A)
    ICD10Code("O80",    "Full-term uncomplicated delivery", "Pregnancy"),
    ICD10Code("O99.419", "Obesity complicating pregnancy", "Pregnancy", is_cc=True),

    # Chapter 18: Symptoms, signs (R00-R99)
    ICD10Code("R07.9",  "Chest pain, unspecified", "Symptoms"),
    ICD10Code("R10.9",  "Abdominal pain, unspecified", "Symptoms"),
    ICD10Code("R11.2",  "Nausea with vomiting", "Symptoms"),
    ICD10Code("R53.83", "Other fatigue", "Symptoms"),
    ICD10Code("R63.4",  "Abnormal weight loss", "Symptoms"),

    # Chapter 19: Injury, poisoning (S00-T88)
    ICD10Code("S72.001A", "Fracture of right femoral neck, initial", "Injury", is_cc=True),
    ICD10Code("S72.141A", "Displaced intertrochanteric fx right femur, init", "Injury", is_cc=True),
    ICD10Code("S06.9X0A", "Unspecified intracranial injury w/o LOC, init", "Injury", is_mcc=True),
    ICD10Code("S82.201A", "Unspecified fracture of right tibia, initial", "Injury"),
    ICD10Code("T81.4XXA", "Infection following a procedure, initial", "Injury", is_cc=True),

    # Chapter 21: Factors influencing health (Z00-Z99)
    ICD10Code("Z00.00", "General adult medical exam", "Health factors"),
    ICD10Code("Z23",    "Encounter for immunization", "Health factors"),
    ICD10Code("Z79.4",  "Long-term insulin use", "Health factors", is_chronic=True),
    ICD10Code("Z79.01", "Long-term anticoagulant use", "Health factors", is_chronic=True),
    ICD10Code("Z79.899", "Other long-term drug therapy", "Health factors", is_chronic=True),
    ICD10Code("Z66",    "Do-not-resuscitate status", "Health factors"),
    ICD10Code("Z68.41", "BMI 40.0-44.9, adult", "Health factors", is_cc=True),
]


# ── Lookup helpers ─────────────────────────────────────────────────────────

_CODE_MAP: dict[str, ICD10Code] = {c.code: c for c in ICD10_CODES}

def lookup(code: str) -> Optional[ICD10Code]:
    """Look up an ICD-10 code in the shipped vocabulary."""
    return _CODE_MAP.get(code)

def by_chapter(chapter: str) -> list[ICD10Code]:
    """Return all codes in a given chapter."""
    return [c for c in ICD10_CODES if c.chapter == chapter]

def chronic_codes() -> list[ICD10Code]:
    """Return all chronic-condition codes."""
    return [c for c in ICD10_CODES if c.is_chronic]

def cc_codes() -> list[ICD10Code]:
    """Return all complication/comorbidity codes."""
    return [c for c in ICD10_CODES if c.is_cc]

def mcc_codes() -> list[ICD10Code]:
    """Return all major CC codes."""
    return [c for c in ICD10_CODES if c.is_mcc]

def all_codes() -> list[str]:
    """Return all shipped ICD-10 code strings."""
    return [c.code for c in ICD10_CODES]
