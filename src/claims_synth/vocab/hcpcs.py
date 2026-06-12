"""
HCPCS Level II codes — public domain subset.

HCPCS Level II codes are published by CMS without copyright restriction.
These are letter-prefixed codes (A, B, E, G, H, J, K, L, M, P, Q, R, S, T, V).
No copyrighted descriptions — only code + short public-domain category label.

CPT codes (Level I, 5-digit numeric) are AMA-copyrighted and NOT included.
See cpt.py for the opaque-ID interface for user-supplied CPT codes.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class HCPCSCode:
    code: str
    category: str          # public-domain short label
    group: str             # HCPCS group
    is_drug: bool = False
    is_dme: bool = False   # durable medical equipment
    is_supply: bool = False
    is_transport: bool = False
    is_temp: bool = False  # temporary G/Q code
    mue_units: Optional[int] = None  # Medically Unlikely Edit max units


# ── Curated HCPCS Level II subset ──────────────────────────────────────────
# Only letter-prefixed codes. Frequencies approximate CMS Medicare data.
# CPT descriptions NOT included.

HCPCS_CODES: list[HCPCSCode] = [
    # ── Drugs / Biologicals (J-codes) ──────────────────────────────────────
    HCPCSCode("J0135", "Adalimumab injection, 20mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J0171", "Adrenalin injection, 0.1mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J0585", "OnabotulinumtoxinA, 1 unit", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J0696", "Ceftriaxone injection, 250mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J0897", "Denosumab injection, 1mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J1100", "Dexamethasone injection, 1mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J1745", "Infliximab injection, 10mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J1885", "Ketorolac injection, 15mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J2357", "Omalizumab injection, 5mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J2505", "Pegfilgrastim injection, 6mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J2778", "Ranibizumab injection, 0.1mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J3262", "Tocilizumab injection, 1mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J3420", "Vitamin B12 injection, up to 1000mcg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J7050", "Normal saline infusion, 250ml", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J7120", "Ringers lactate infusion, up to 1000ml", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J9263", "Oxaliplatin injection, 0.5mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J9310", "Rituximab injection, 100mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J9355", "Trastuzumab injection, 10mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J9035", "Bevacizumab injection, 10mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J9041", "Bortezomib injection, 0.1mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J9171", "Docetaxel injection, 1mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J9190", "Fluorouracil injection, 500mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J9206", "Irinotecan injection, 20mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J9299", "Nivolumab injection, 1mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J9306", "Pertuzumab injection, 1mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J0894", "Decitabine injection, 1mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J1442", "Filgrastim injection, 1mcg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J1453", "Fosphenytoin injection, 50mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J1642", "Heparin sodium injection, 10 units", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J1650", "Enoxaparin sodium injection, 10mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J2405", "Ondansetron injection, 1mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J3010", "Fentanyl citrate injection, 0.1mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J3370", "Vancomycin injection, 500mg", "Drugs", is_drug=True, mue_units=1),
    HCPCSCode("J3485", "Zidovudine injection, 10mg", "Drugs", is_drug=True, mue_units=1),

    # ── DME (E-codes, K-codes) ─────────────────────────────────────────────
    HCPCSCode("E0100", "Cane, adjustable", "DME", is_dme=True, mue_units=1),
    HCPCSCode("E0141", "Walker, rigid, wheeled", "DME", is_dme=True, mue_units=1),
    HCPCSCode("E0143", "Walker, folding, wheeled", "DME", is_dme=True, mue_units=1),
    HCPCSCode("E0260", "Hospital bed, semi-electric", "DME", is_dme=True, mue_units=1),
    HCPCSCode("E0270", "Hospital bed, heavy duty", "DME", is_dme=True, mue_units=1),
    HCPCSCode("E0601", "CPAP device", "DME", is_dme=True, mue_units=1),
    HCPCSCode("E1390", "Oxygen concentrator", "DME", is_dme=True, mue_units=1),
    HCPCSCode("E1392", "Oxygen concentrator, portable", "DME", is_dme=True, mue_units=1),
    HCPCSCode("K0001", "Standard wheelchair", "DME", is_dme=True, mue_units=1),
    HCPCSCode("K0004", "Lightweight wheelchair", "DME", is_dme=True, mue_units=1),
    HCPCSCode("K0011", "Power wheelchair, standard", "DME", is_dme=True, mue_units=1),
    HCPCSCode("K0108", "Wheelchair component, not otherwise specified", "DME", is_dme=True, mue_units=1),
    HCPCSCode("K0733", "Power wheelchair, group 2 standard, portable", "DME", is_dme=True, mue_units=1),

    # ── Supplies (A-codes) ─────────────────────────────────────────────────
    HCPCSCode("A4253", "Blood glucose test strips, 50/box", "Supplies", is_supply=True, mue_units=1),
    HCPCSCode("A4256", "Blood glucose monitor with integrated lancing", "Supplies", is_supply=True, mue_units=1),
    HCPCSCode("A4258", "Blood glucose test strips, 100/box", "Supplies", is_supply=True, mue_units=1),
    HCPCSCode("A9270", "Non-covered item (patient responsibility)", "Supplies", is_supply=True, mue_units=1),
    HCPCSCode("A4335", "Incontinence supply, miscellaneous", "Supplies", is_supply=True, mue_units=1),
    HCPCSCode("A4351", "Intermittent urinary catheter, straight tip", "Supplies", is_supply=True, mue_units=1),
    HCPCSCode("A4450", "Tape, non-waterproof, per 18 sq inches", "Supplies", is_supply=True, mue_units=1),
    HCPCSCode("A4452", "Tape, waterproof, per 18 sq inches", "Supplies", is_supply=True, mue_units=1),
    HCPCSCode("A4550", "Surgical tray", "Supplies", is_supply=True, mue_units=1),
    HCPCSCode("A4649", "Surgical supply, miscellaneous", "Supplies", is_supply=True, mue_units=1),
    HCPCSCode("A4927", "Gloves, non-sterile, per pair", "Supplies", is_supply=True, mue_units=1),
    HCPCSCode("A4930", "Gloves, sterile, per pair", "Supplies", is_supply=True, mue_units=1),
    HCPCSCode("A5500", "Diabetic shoe, per pair", "Supplies", is_supply=True, mue_units=1),
    HCPCSCode("A6010", "Collagen based wound filler, per gram", "Supplies", is_supply=True, mue_units=1),
    HCPCSCode("A6196", "Alginate wound dressing, sterile, pad", "Supplies", is_supply=True, mue_units=1),
    HCPCSCode("A6197", "Alginate wound dressing, sterile, 16 sq in or less", "Supplies", is_supply=True, mue_units=1),
    HCPCSCode("A6203", "Composite wound dressing, sterile, pad", "Supplies", is_supply=True, mue_units=1),
    HCPCSCode("A6209", "Foam wound dressing, sterile, pad", "Supplies", is_supply=True, mue_units=1),
    HCPCSCode("A6212", "Foam wound dressing, sterile, 16 sq in or less", "Supplies", is_supply=True, mue_units=1),
    HCPCSCode("A6402", "Gauze, sterile, pad, 16 sq in or less", "Supplies", is_supply=True, mue_units=1),
    HCPCSCode("A6404", "Gauze, sterile, per sq yd up to 16 sq in", "Supplies", is_supply=True, mue_units=1),

    # ── Ambulance / Transport (A-codes) ────────────────────────────────────
    HCPCSCode("A0426", "ALS ambulance, level 1, non-emergency", "Transport", is_transport=True, mue_units=1),
    HCPCSCode("A0427", "ALS ambulance, level 1, emergency", "Transport", is_transport=True, mue_units=1),
    HCPCSCode("A0428", "BLS ambulance, non-emergency", "Transport", is_transport=True, mue_units=1),
    HCPCSCode("A0429", "BLS ambulance, emergency", "Transport", is_transport=True, mue_units=1),
    HCPCSCode("A0430", "ALS ambulance, level 2, emergency", "Transport", is_transport=True, mue_units=1),
    HCPCSCode("A0431", "ALS ambulance, level 2, non-emergency", "Transport", is_transport=True, mue_units=1),
    HCPCSCode("A0433", "ALS ambulance, specialty care transport", "Transport", is_transport=True, mue_units=1),
    HCPCSCode("A0434", "ALS ambulance, specialty care transport, level 2", "Transport", is_transport=True, mue_units=1),
    HCPCSCode("A0999", "Ambulance service, unlisted", "Transport", is_transport=True, mue_units=1),

    # ── Temporary / Procedure codes (G-codes, Q-codes) ─────────────────────
    # CMS temporary codes for services not yet in CPT. Public domain.
    HCPCSCode("G0008", "Administration of influenza vaccine", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("G0009", "Administration of pneumococcal vaccine", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("G0101", "Cervical/vaginal cancer screening, pelvic/breast exam", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("G0105", "Colorectal cancer screening, colonoscopy, high risk", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("G0121", "Colorectal cancer screening, colonoscopy, not high risk", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("G0202", "Screening mammography, digital, bilateral", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("G0204", "Diagnostic mammography, digital, bilateral", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("G0270", "Medical nutrition therapy, individual, 15 min", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("G0402", "Initial preventive physical exam, IPPE", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("G0438", "Annual wellness visit, initial", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("G0439", "Annual wellness visit, subsequent", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("G0444", "Depression screening, annual, 15 min", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("G0447", "Behavioral counseling for obesity, 15 min", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("G0463", "Hospital outpatient clinic visit for assessment", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("G0471", "Collection of venous blood by venipuncture", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("G0472", "Hepatitis C antibody screening, individual", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("G2010", "Remote evaluation of recorded video/images, established patient", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("G2012", "Virtual check-in, 5-10 min", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("G2061", "Qualified non-physician healthcare professional online assessment, 5-10 min", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("G2062", "Qualified non-physician healthcare professional online assessment, 11-20 min", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("G2063", "Qualified non-physician healthcare professional online assessment, 21+ min", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("G2250", "Remote therapeutic monitoring, initial set-up", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("G2251", "Remote therapeutic monitoring, supply of device", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("G2252", "Remote therapeutic monitoring, treatment management, 20 min", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("Q0091", "Screening Papanicolaou smear, obtaining, preparing and conveyance", "Temp", is_temp=True, mue_units=1),
    HCPCSCode("Q3014", "Telehealth originating site facility fee", "Temp", is_temp=True, mue_units=1),

    # ── Modifiers (informational — not billable alone) ─────────────────────
    HCPCSCode("25", "Significant, separately identifiable E/M service", "Modifier", mue_units=None),
    HCPCSCode("26", "Professional component", "Modifier", mue_units=None),
    HCPCSCode("50", "Bilateral procedure", "Modifier", mue_units=None),
    HCPCSCode("51", "Multiple procedures", "Modifier", mue_units=None),
    HCPCSCode("59", "Distinct procedural service", "Modifier", mue_units=None),
    HCPCSCode("76", "Repeat procedure by same physician", "Modifier", mue_units=None),
    HCPCSCode("77", "Repeat procedure by another physician", "Modifier", mue_units=None),
    HCPCSCode("LT", "Left side", "Modifier", mue_units=None),
    HCPCSCode("RT", "Right side", "Modifier", mue_units=None),
    HCPCSCode("XE", "Separate encounter", "Modifier", mue_units=None),
    HCPCSCode("XS", "Separate structure", "Modifier", mue_units=None),
    HCPCSCode("XP", "Separate practitioner", "Modifier", mue_units=None),
    HCPCSCode("XU", "Unusual non-overlapping service", "Modifier", mue_units=None),
]


# ── Lookup helpers ─────────────────────────────────────────────────────────

_CODE_MAP: dict[str, HCPCSCode] = {c.code: c for c in HCPCS_CODES}

def lookup(code: str) -> Optional[HCPCSCode]:
    """Look up an HCPCS code in the shipped vocabulary."""
    return _CODE_MAP.get(code)

def by_group(group: str) -> list[HCPCSCode]:
    """Return all codes in a given group."""
    return [c for c in HCPCS_CODES if c.group == group]

def drug_codes() -> list[HCPCSCode]:
    return [c for c in HCPCS_CODES if c.is_drug]

def dme_codes() -> list[HCPCSCode]:
    return [c for c in HCPCS_CODES if c.is_dme]

def supply_codes() -> list[HCPCSCode]:
    return [c for c in HCPCS_CODES if c.is_supply]

def transport_codes() -> list[HCPCSCode]:
    return [c for c in HCPCS_CODES if c.is_transport]

def temp_codes() -> list[HCPCSCode]:
    return [c for c in HCPCS_CODES if c.is_temp]

def modifier_codes() -> list[HCPCSCode]:
    return [c for c in HCPCS_CODES if c.group == "Modifier"]

def billable_codes() -> list[HCPCSCode]:
    """All codes except modifiers (modifiers aren't billable alone)."""
    return [c for c in HCPCS_CODES if c.group != "Modifier"]

def all_codes() -> list[str]:
    """Return all shipped HCPCS code strings."""
    return [c.code for c in HCPCS_CODES]
