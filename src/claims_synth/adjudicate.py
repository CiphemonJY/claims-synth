"""
Rules-based payer adjudicator for claims-synth.

Takes a synthetic Claim (837) and produces an AdjudicationResult (835-like
remittance advice) with per-line payment decisions, adjustments, and
standard reason codes.

All reason codes are from the CMS public-domain CARC/RARC lists.
Zero copyrighted content.

Design:
  - Each payer has a profile (denial rates, strictness, bundling rules)
  - Rules are applied deterministically by seed
  - Output: paid amount, denial reason, adjustment codes per line
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from enum import Enum
from typing import Optional

import numpy as np

from claims_synth.claim import Claim, ClaimLine, ClaimType, PlaceOfService, Payer
from claims_synth.vocab.icd10 import lookup as icd10_lookup
from claims_synth.vocab.hcpcs import lookup as hcpcs_lookup


# ── Reason Codes (CMS public domain) ────────────────────────────────────────
# CARC = Claim Adjustment Reason Code
# RARC = Remittance Advice Remark Code
# Group codes: CO (Contractual Obligation), PR (Patient Responsibility),
#              OA (Other Adjustment), PI (Payer Initiated)

class GroupCode(str, Enum):
    CO = "CO"  # Contractual Obligation — provider write-off
    PR = "PR"  # Patient Responsibility — patient owes
    OA = "OA"  # Other Adjustment
    PI = "PI"  # Payer Initiated Reduction


@dataclass
class ReasonCode:
    """A CARC or RARC reason code (CMS public domain)."""
    code: str           # e.g. "16", "M51", "N130"
    type: str           # "CARC" or "RARC"
    group: GroupCode
    description: str    # public-domain short label


# ── Common CARC codes ───────────────────────────────────────────────────────
# From CMS X12 835 TR3 — public domain

CARC_CODES: dict[str, ReasonCode] = {
    "16": ReasonCode("16", "CARC", GroupCode.CO,
                     "Claim/service lacks information for adjudication"),
    "18": ReasonCode("18", "CARC", GroupCode.CO,
                     "Exact duplicate claim/service"),
    "22": ReasonCode("22", "CARC", GroupCode.CO,
                     "Payment adjusted because this care may be covered by another payer"),
    "29": ReasonCode("29", "CARC", GroupCode.PR,
                     "The time limit for filing has expired"),
    "31": ReasonCode("31", "CARC", GroupCode.PR,
                     "Patient cannot be identified as our insured"),
    "50": ReasonCode("50", "CARC", GroupCode.CO,
                     "These are non-covered services"),
    "96": ReasonCode("96", "CARC", GroupCode.CO,
                     "Non-covered charge(s)"),
    "97": ReasonCode("97", "CARC", GroupCode.CO,
                     "Payment included in allowance for another service"),
    "109": ReasonCode("109", "CARC", GroupCode.CO,
                      "Claim/service not covered by this payer/contractor"),
    "119": ReasonCode("119", "CARC", GroupCode.CO,
                      "Benefit maximum for this time period has been reached"),
    "140": ReasonCode("140", "CARC", GroupCode.PR,
                      "Patient/Insured health identification number not valid"),
    "151": ReasonCode("151", "CARC", GroupCode.PR,
                      "Payment adjusted because the payer deems the information submitted "
                      "does not support this level of service"),
    "167": ReasonCode("167", "CARC", GroupCode.CO,
                      "These diagnosis(es) are not covered"),
    "170": ReasonCode("170", "CARC", GroupCode.CO,
                      "Payment is denied when performed/billed by this type of provider"),
    "177": ReasonCode("177", "CARC", GroupCode.PR,
                      "Patient has not met the required eligibility requirements"),
    "197": ReasonCode("197", "CARC", GroupCode.CO,
                      "Precertification/authorization absent"),
    "198": ReasonCode("198", "CARC", GroupCode.CO,
                      "Precertification/authorization exceeded"),
    "204": ReasonCode("204", "CARC", GroupCode.CO,
                      "This service/equipment/drug is not covered under the "
                      "patient's current benefit plan"),
    "234": ReasonCode("234", "CARC", GroupCode.CO,
                      "Procedure is not paid separately"),
    "236": ReasonCode("236", "CARC", GroupCode.CO,
                      "This procedure is not paid separately"),
    "B15": ReasonCode("B15", "CARC", GroupCode.CO,
                      "Payment adjusted because this service/procedure is not paid "
                      "when performed in this place of service"),
    "B22": ReasonCode("B22", "CARC", GroupCode.CO,
                      "Payment denied because this procedure code/modifier was "
                      "invalid on the date of service"),
}

# ── Common RARC codes ───────────────────────────────────────────────────────
RARC_CODES: dict[str, ReasonCode] = {
    "M51": ReasonCode("M51", "RARC", GroupCode.CO,
                      "Missing/incomplete/invalid procedure code(s)"),
    "M81": ReasonCode("M81", "RARC", GroupCode.CO,
                      "You must provide the appropriate primary diagnosis code"),
    "N130": ReasonCode("N130", "RARC", GroupCode.CO,
                       "Consult plan benefit documents/guidelines for information "
                       "about restrictions for this service"),
    "N174": ReasonCode("N174", "RARC", GroupCode.CO,
                       "This is not a covered service/procedure in this setting"),
    "N362": ReasonCode("N362", "RARC", GroupCode.CO,
                       "The diagnosis code is inconsistent with the procedure code"),
    "N382": ReasonCode("N382", "RARC", GroupCode.CO,
                       "Missing/incomplete/invalid patient identifier"),
    "N425": ReasonCode("N425", "RARC", GroupCode.PR,
                       "Statutorily excluded service(s)"),
    "N519": ReasonCode("N519", "RARC", GroupCode.CO,
                       "Invalid combination of HCPCS modifiers"),
    "N574": ReasonCode("N574", "RARC", GroupCode.CO,
                       "The clinical information does not support the medical "
                       "necessity for this service"),
    "N657": ReasonCode("N657", "RARC", GroupCode.CO,
                       "This service should be billed with the appropriate "
                       "code for that service"),
}


# ── Payer Profiles ──────────────────────────────────────────────────────────

@dataclass
class PayerProfile:
    """Adjudication behavior for a specific payer."""
    payer_id: str
    plan_type: str

    # Base denial rates (probability a line is denied for any reason)
    denial_rate: float = 0.15          # overall line denial probability

    # Rule-specific denial probabilities (conditional on the rule triggering)
    auth_denial_rate: float = 0.90      # deny if auth missing/denied
    timely_filing_days: int = 180       # days from service to claim receipt
    timely_filing_denial_rate: float = 0.95

    # Medical necessity strictness (0.0 = never deny, 1.0 = always deny on mismatch)
    medical_necessity_strictness: float = 0.30

    # Bundling: probability of bundling related procedures
    bundling_rate: float = 0.10

    # Place of service restrictions
    pos_restrictions: bool = True       # deny if POS doesn't match procedure

    # Payment rate (fraction of charge paid when approved)
    payment_rate: float = 0.80         # Medicare-like: 80% of allowable

    # Patient responsibility share
    patient_share: float = 0.20        # coinsurance

    # Deductible (applied per claim)
    deductible: float = 0.0


# ── Payer profiles keyed by opaque payer ID ─────────────────────────────────

PAYER_PROFILES: dict[str, PayerProfile] = {
    "PAYER_A": PayerProfile(
        payer_id="PAYER_A", plan_type="COMMERCIAL",
        denial_rate=0.12, auth_denial_rate=0.85,
        timely_filing_days=180, medical_necessity_strictness=0.25,
        bundling_rate=0.08, pos_restrictions=True,
        payment_rate=0.80, patient_share=0.20, deductible=500.0,
    ),
    "PAYER_B": PayerProfile(
        payer_id="PAYER_B", plan_type="COMMERCIAL",
        denial_rate=0.10, auth_denial_rate=0.80,
        timely_filing_days=365, medical_necessity_strictness=0.20,
        bundling_rate=0.05, pos_restrictions=True,
        payment_rate=0.85, patient_share=0.15, deductible=250.0,
    ),
    "PAYER_C": PayerProfile(
        payer_id="PAYER_C", plan_type="MEDICARE",
        denial_rate=0.08, auth_denial_rate=0.95,
        timely_filing_days=365, medical_necessity_strictness=0.40,
        bundling_rate=0.15, pos_restrictions=True,
        payment_rate=0.80, patient_share=0.20, deductible=0.0,
    ),
    "PAYER_D": PayerProfile(
        payer_id="PAYER_D", plan_type="MEDICARE_ADVANTAGE",
        denial_rate=0.14, auth_denial_rate=0.90,
        timely_filing_days=365, medical_necessity_strictness=0.35,
        bundling_rate=0.12, pos_restrictions=True,
        payment_rate=0.80, patient_share=0.20, deductible=0.0,
    ),
    "PAYER_E": PayerProfile(
        payer_id="PAYER_E", plan_type="MEDICAID",
        denial_rate=0.10, auth_denial_rate=0.85,
        timely_filing_days=365, medical_necessity_strictness=0.30,
        bundling_rate=0.10, pos_restrictions=True,
        payment_rate=0.70, patient_share=0.00, deductible=0.0,
    ),
    "PAYER_F": PayerProfile(
        payer_id="PAYER_F", plan_type="COMMERCIAL",
        denial_rate=0.18, auth_denial_rate=0.92,
        timely_filing_days=90, medical_necessity_strictness=0.45,
        bundling_rate=0.15, pos_restrictions=True,
        payment_rate=0.75, patient_share=0.25, deductible=1000.0,
    ),
    "PAYER_G": PayerProfile(
        payer_id="PAYER_G", plan_type="MEDICARE_ADVANTAGE",
        denial_rate=0.16, auth_denial_rate=0.88,
        timely_filing_days=365, medical_necessity_strictness=0.38,
        bundling_rate=0.14, pos_restrictions=True,
        payment_rate=0.80, patient_share=0.20, deductible=0.0,
    ),
    "PAYER_H": PayerProfile(
        payer_id="PAYER_H", plan_type="MEDICAID",
        denial_rate=0.12, auth_denial_rate=0.82,
        timely_filing_days=365, medical_necessity_strictness=0.28,
        bundling_rate=0.08, pos_restrictions=True,
        payment_rate=0.65, patient_share=0.00, deductible=0.0,
    ),
}


# ── Adjudication Result ─────────────────────────────────────────────────────

@dataclass
class LineAdjudication:
    """Adjudication decision for a single claim line."""
    sequence: int                      # matches ClaimLine.sequence
    procedure_code: str
    status: str                        # "paid", "denied", "adjusted", "bundled"
    submitted_charge: float
    allowed_amount: float = 0.0
    paid_amount: float = 0.0
    patient_responsibility: float = 0.0
    adjustments: list[dict] = field(default_factory=list)
    # Each adjustment: {"group": "CO", "code": "16", "amount": 100.0, "reason": "..."}
    denial_reasons: list[str] = field(default_factory=list)  # human-readable


@dataclass
class AdjudicationResult:
    """Complete 835-like remittance for a claim."""
    claim_id: str
    payer_id: str
    payer_profile: str                 # plan_type
    adjudication_date: str             # ISO date
    claim_status: str                  # "paid", "partially_paid", "fully_denied"
    total_submitted: float = 0.0
    total_allowed: float = 0.0
    total_paid: float = 0.0
    total_patient_responsibility: float = 0.0
    total_adjustments: float = 0.0
    lines: list[LineAdjudication] = field(default_factory=list)
    claim_level_adjustments: list[dict] = field(default_factory=list)
    check_number: str = ""
    payment_date: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @property
    def denial_rate(self) -> float:
        """Fraction of lines denied."""
        if not self.lines:
            return 0.0
        denied = sum(1 for l in self.lines if l.status == "denied")
        return denied / len(self.lines)

    @property
    def payment_rate(self) -> float:
        """Fraction of submitted charges actually paid."""
        if self.total_submitted <= 0:
            return 0.0
        return self.total_paid / self.total_submitted


# ── Diagnosis-Procedure compatibility matrix ────────────────────────────────
# Maps ICD-10 chapters to HCPCS groups that are medically plausible.
# Anything outside this matrix triggers a medical necessity review.

DX_PROC_COMPAT: dict[str, set[str]] = {
    "Infectious":       {"Drugs", "Temp", "Supplies"},
    "Neoplasms":        {"Drugs", "Temp", "Supplies", "DME"},
    "Blood":            {"Drugs", "Temp", "Supplies"},
    "Endocrine":        {"Drugs", "Temp", "Supplies", "DME"},
    "Mental":           {"Drugs", "Temp"},
    "Nervous":          {"Drugs", "Temp", "Supplies", "DME", "Transport"},
    "Eye":              {"Drugs", "Temp", "Supplies"},
    "Circulatory":      {"Drugs", "Temp", "Supplies", "DME", "Transport"},
    "Respiratory":      {"Drugs", "Temp", "Supplies", "DME", "Transport"},
    "Digestive":        {"Drugs", "Temp", "Supplies"},
    "Skin":             {"Drugs", "Temp", "Supplies"},
    "Musculoskeletal":  {"Drugs", "Temp", "Supplies", "DME", "Transport"},
    "Genitourinary":    {"Drugs", "Temp", "Supplies"},
    "Pregnancy":        {"Drugs", "Temp", "Supplies", "Transport"},
    "Symptoms":         {"Drugs", "Temp", "Supplies", "Transport"},
    "Injury":           {"Drugs", "Temp", "Supplies", "DME", "Transport"},
    "Health factors":   {"Temp", "Supplies"},
}


# ── Place of Service restrictions ───────────────────────────────────────────
# Some HCPCS groups are only payable in specific settings

POS_RESTRICTIONS: dict[str, set[str]] = {
    "Drugs":     {"OFFICE", "OUTPATIENT", "INPATIENT", "EMERGENCY", "ASC"},
    "DME":       {"OFFICE", "OUTPATIENT"},
    "Supplies":  {"OFFICE", "OUTPATIENT", "INPATIENT", "EMERGENCY", "ASC"},
    "Transport": {"EMERGENCY", "INPATIENT"},
    "Temp":      {"OFFICE", "OUTPATIENT", "INPATIENT", "EMERGENCY", "ASC", "TELEHEALTH"},
}


# ── Bundling rules ───────────────────────────────────────────────────────────
# Pairs of HCPCS groups that are commonly bundled (the lower-charge line
# gets absorbed into the higher-charge line)

BUNDLE_PAIRS: list[tuple[str, str]] = [
    ("Temp", "Temp"),       # E/M + procedure
    ("Drugs", "Supplies"),  # drug + administration supplies
    ("Supplies", "Temp"),   # supplies bundled into procedure
]


# ── Adjudicator ─────────────────────────────────────────────────────────────

class Adjudicator:
    """
    Rules-based payer adjudicator.

    Deterministic by seed — same claim + same seed = same result.
    Applies payer-specific profiles for denial rates, medical necessity
    strictness, bundling, timely filing, and place-of-service rules.

    Usage:
        adj = Adjudicator(seed=42)
        result = adj.adjudicate(claim)
        print(result.to_json())
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self._check_counter = 0

    def adjudicate(self, claim: Claim,
                   adjudication_date: Optional[date] = None) -> AdjudicationResult:
        """
        Adjudicate a single claim.

        Args:
            claim: The Claim to adjudicate.
            adjudication_date: Date of adjudication (default: today).

        Returns:
            AdjudicationResult with per-line decisions.
        """
        if adjudication_date is None:
            adjudication_date = date.today()

        # ── Get payer profile ──────────────────────────────────────────
        payer = claim.payer
        if payer is None:
            raise ValueError("Claim has no payer — cannot adjudicate")

        profile = PAYER_PROFILES.get(payer.payer_id)
        if profile is None:
            # Unknown payer — use a default commercial profile
            profile = PayerProfile(
                payer_id=payer.payer_id,
                plan_type=payer.plan_type or "COMMERCIAL",
            )

        # ── Claim-level checks ──────────────────────────────────────────
        claim_adjustments: list[dict] = []
        claim_denied = False

        # Timely filing
        if claim.statement_to:
            days_since = (adjudication_date - claim.statement_to).days
            if days_since > profile.timely_filing_days:
                if self.rng.random() < profile.timely_filing_denial_rate:
                    claim_denied = True
                    claim_adjustments.append({
                        "group": "PR", "code": "29", "amount": claim.total_charge,
                        "reason": f"Timely filing expired ({days_since}d > "
                                  f"{profile.timely_filing_days}d limit)",
                    })

        # ── Per-line adjudication ──────────────────────────────────────
        line_results: list[LineAdjudication] = []
        bundled_indices: set[int] = set()

        # Check for bundling opportunities
        if len(claim.lines) >= 2:
            bundled_indices = self._apply_bundling(claim, profile)

        for line in claim.lines:
            result = self._adjudicate_line(
                claim, line, profile, adjudication_date,
                claim_denied, bundled_indices,
            )
            line_results.append(result)

        # ── Compute totals ─────────────────────────────────────────────
        total_submitted = claim.total_charge
        total_allowed = sum(l.allowed_amount for l in line_results)
        total_paid = sum(l.paid_amount for l in line_results)
        total_pr = sum(l.patient_responsibility for l in line_results)
        total_adjustments = sum(
            sum(adj.get("amount", 0) for adj in l.adjustments)
            for l in line_results
        ) + sum(adj.get("amount", 0) for adj in claim_adjustments)

        # ── Claim status ───────────────────────────────────────────────
        if all(l.status == "denied" for l in line_results):
            claim_status = "fully_denied"
        elif any(l.status == "denied" for l in line_results):
            claim_status = "partially_paid"
        else:
            claim_status = "paid"

        # ── Payment info ───────────────────────────────────────────────
        self._check_counter += 1
        check_number = f"CK{self._check_counter:08d}"
        payment_date = adjudication_date.isoformat()

        return AdjudicationResult(
            claim_id=claim.claim_id,
            payer_id=profile.payer_id,
            payer_profile=profile.plan_type,
            adjudication_date=adjudication_date.isoformat(),
            claim_status=claim_status,
            total_submitted=round(total_submitted, 2),
            total_allowed=round(total_allowed, 2),
            total_paid=round(total_paid, 2),
            total_patient_responsibility=round(total_pr, 2),
            total_adjustments=round(total_adjustments, 2),
            lines=line_results,
            claim_level_adjustments=claim_adjustments,
            check_number=check_number,
            payment_date=payment_date,
        )

    def _adjudicate_line(
        self,
        claim: Claim,
        line: ClaimLine,
        profile: PayerProfile,
        adjudication_date: date,
        claim_denied: bool,
        bundled_indices: set[int],
    ) -> LineAdjudication:
        """Adjudicate a single claim line."""

        adjustments: list[dict] = []
        denial_reasons: list[str] = []
        status = "paid"
        allowed = line.charge * line.units

        # ── Claim-level denial propagates ──────────────────────────────
        if claim_denied:
            status = "denied"
            denial_reasons.append("Claim-level denial (timely filing)")
            adjustments.append({
                "group": "PR", "code": "29",
                "amount": allowed,
                "reason": "Timely filing expired",
            })
            return LineAdjudication(
                sequence=line.sequence,
                procedure_code=line.procedure_code,
                status=status,
                submitted_charge=allowed,
                allowed_amount=0.0,
                paid_amount=0.0,
                patient_responsibility=allowed,
                adjustments=adjustments,
                denial_reasons=denial_reasons,
            )

        # ── Bundled ─────────────────────────────────────────────────────
        if line.sequence in bundled_indices:
            status = "bundled"
            denial_reasons.append("Bundled into primary procedure")
            adjustments.append({
                "group": "CO", "code": "97",
                "amount": allowed,
                "reason": "Payment included in allowance for another service",
            })
            return LineAdjudication(
                sequence=line.sequence,
                procedure_code=line.procedure_code,
                status=status,
                submitted_charge=allowed,
                allowed_amount=0.0,
                paid_amount=0.0,
                patient_responsibility=0.0,
                adjustments=adjustments,
                denial_reasons=denial_reasons,
            )

        # ── Authorization check ────────────────────────────────────────
        auth = claim.authorization
        if auth and auth.auth_status == "denied":
            if self.rng.random() < profile.auth_denial_rate:
                status = "denied"
                denial_reasons.append("Authorization denied")
                adjustments.append({
                    "group": "CO", "code": "198",
                    "amount": allowed,
                    "reason": "Precertification/authorization exceeded",
                })
                return LineAdjudication(
                    sequence=line.sequence,
                    procedure_code=line.procedure_code,
                    status=status,
                    submitted_charge=allowed,
                    allowed_amount=0.0,
                    paid_amount=0.0,
                    patient_responsibility=0.0,
                    adjustments=adjustments,
                    denial_reasons=denial_reasons,
                )

        if auth and auth.auth_number is None and auth.auth_status != "not_required":
            if self.rng.random() < profile.auth_denial_rate * 0.5:
                status = "denied"
                denial_reasons.append("Missing authorization")
                adjustments.append({
                    "group": "CO", "code": "197",
                    "amount": allowed,
                    "reason": "Precertification/authorization absent",
                })

        # ── Medical necessity check ────────────────────────────────────
        if status == "paid":
            status, adjustments, denial_reasons = self._check_medical_necessity(
                claim, line, profile, status, adjustments, denial_reasons, allowed,
            )

        # ── Place of service check ─────────────────────────────────────
        if status == "paid" and profile.pos_restrictions:
            status, adjustments, denial_reasons = self._check_pos(
                claim, line, profile, status, adjustments, denial_reasons, allowed,
            )

        # ── Random denial (payer-specific base rate) ────────────────────
        if status == "paid":
            if self.rng.random() < profile.denial_rate:
                status = "denied"
                denial_reasons.append("Payer-specific denial")
                adjustments.append({
                    "group": "CO", "code": "204",
                    "amount": allowed,
                    "reason": "Not covered under patient's current benefit plan",
                })

        # ── Compute payment ────────────────────────────────────────────
        if status == "paid":
            # Apply deductible
            remaining = allowed
            if profile.deductible > 0:
                deductible_applied = min(profile.deductible, remaining)
                remaining -= deductible_applied
                if deductible_applied > 0:
                    adjustments.append({
                        "group": "PR", "code": "177",
                        "amount": deductible_applied,
                        "reason": "Patient has not met required deductible",
                    })

            # Payer pays its share of the remaining.
            # patient_share=0 means payer pays 100% (Medicaid).
            # Otherwise payer pays (1 - patient_share) of remaining.
            payer_share = 1.0 - profile.patient_share
            payer_paid = round(remaining * payer_share, 2)
            patient_owes = round(remaining - payer_paid, 2)

            return LineAdjudication(
                sequence=line.sequence,
                procedure_code=line.procedure_code,
                status="paid",
                submitted_charge=allowed,
                allowed_amount=allowed,
                paid_amount=payer_paid,
                patient_responsibility=patient_owes,
                adjustments=adjustments,
                denial_reasons=denial_reasons,
            )
        else:
            return LineAdjudication(
                sequence=line.sequence,
                procedure_code=line.procedure_code,
                status=status,
                submitted_charge=allowed,
                allowed_amount=0.0,
                paid_amount=0.0,
                patient_responsibility=allowed,
                adjustments=adjustments,
                denial_reasons=denial_reasons,
            )

    def _check_medical_necessity(
        self,
        claim: Claim,
        line: ClaimLine,
        profile: PayerProfile,
        status: str,
        adjustments: list[dict],
        denial_reasons: list[str],
        allowed: float,
    ) -> tuple[str, list[dict], list[str]]:
        """Check if diagnoses support the procedure."""

        # Get HCPCS group for this line
        hcpcs = hcpcs_lookup(line.procedure_code)
        if hcpcs is None:
            # Unknown code — can't check, let it through
            return status, adjustments, denial_reasons

        proc_group = hcpcs.group
        if proc_group == "Modifier":
            return status, adjustments, denial_reasons

        # Check each diagnosis pointer
        mismatches = 0
        for ptr in line.diagnosis_pointers:
            if ptr < 1 or ptr > len(claim.diagnoses):
                continue
            dx = claim.diagnoses[ptr - 1]
            dx_info = icd10_lookup(dx.code)
            if dx_info is None:
                continue

            compatible_groups = DX_PROC_COMPAT.get(dx_info.chapter, set())
            if proc_group not in compatible_groups:
                mismatches += 1

        # If all pointed diagnoses are incompatible, it's a hard mismatch
        if mismatches > 0 and mismatches == len(line.diagnosis_pointers):
            if self.rng.random() < profile.medical_necessity_strictness:
                status = "denied"
                denial_reasons.append(
                    f"Medical necessity: {proc_group} not supported by "
                    f"diagnosis codes"
                )
                adjustments.append({
                    "group": "CO", "code": "167",
                    "amount": allowed,
                    "reason": "These diagnosis(es) are not covered",
                })

        return status, adjustments, denial_reasons

    def _check_pos(
        self,
        claim: Claim,
        line: ClaimLine,
        profile: PayerProfile,
        status: str,
        adjustments: list[dict],
        denial_reasons: list[str],
        allowed: float,
    ) -> tuple[str, list[dict], list[str]]:
        """Check place of service appropriateness."""

        if claim.encounter is None:
            return status, adjustments, denial_reasons

        hcpcs = hcpcs_lookup(line.procedure_code)
        if hcpcs is None:
            return status, adjustments, denial_reasons

        proc_group = hcpcs.group
        allowed_pos = POS_RESTRICTIONS.get(proc_group, set())
        if not allowed_pos:
            return status, adjustments, denial_reasons

        pos_name = claim.encounter.place_of_service.name
        if pos_name not in allowed_pos:
            if self.rng.random() < 0.80:  # high denial rate for POS mismatch
                status = "denied"
                denial_reasons.append(
                    f"Place of service mismatch: {proc_group} not payable "
                    f"in {pos_name}"
                )
                adjustments.append({
                    "group": "CO", "code": "B15",
                    "amount": allowed,
                    "reason": "Service not paid in this place of service",
                })

        return status, adjustments, denial_reasons

    def _apply_bundling(
        self, claim: Claim, profile: PayerProfile,
    ) -> set[int]:
        """Identify lines to bundle. Returns set of line sequences to bundle."""

        bundled: set[int] = set()

        for i, line_a in enumerate(claim.lines):
            for j, line_b in enumerate(claim.lines):
                if j <= i:
                    continue
                if line_a.sequence in bundled or line_b.sequence in bundled:
                    continue

                hcpcs_a = hcpcs_lookup(line_a.procedure_code)
                hcpcs_b = hcpcs_lookup(line_b.procedure_code)
                if hcpcs_a is None or hcpcs_b is None:
                    continue

                group_a = hcpcs_a.group
                group_b = hcpcs_b.group

                # Check if this pair is a bundling candidate
                is_bundle_pair = False
                for (g1, g2) in BUNDLE_PAIRS:
                    if (group_a == g1 and group_b == g2) or \
                       (group_a == g2 and group_b == g1):
                        is_bundle_pair = True
                        break

                if not is_bundle_pair:
                    continue

                if self.rng.random() < profile.bundling_rate:
                    # Bundle the lower-charge line into the higher
                    charge_a = line_a.charge * line_a.units
                    charge_b = line_b.charge * line_b.units
                    if charge_a <= charge_b:
                        bundled.add(line_a.sequence)
                    else:
                        bundled.add(line_b.sequence)

        return bundled


# ── Batch adjudication ──────────────────────────────────────────────────────

def adjudicate_batch(
    claims: list[Claim],
    seed: int = 42,
    adjudication_date: Optional[date] = None,
) -> list[AdjudicationResult]:
    """
    Adjudicate a batch of claims with a single seed.

    Each claim gets a fresh RNG state derived from the seed + claim index,
    so results are deterministic and reproducible.
    """
    results = []
    for i, claim in enumerate(claims):
        adj = Adjudicator(seed=seed + i)
        result = adj.adjudicate(claim, adjudication_date=adjudication_date)
        results.append(result)
    return results


# ── Summary statistics ─────────────────────────────────────────────────────

def summarize(results: list[AdjudicationResult]) -> dict:
    """Compute aggregate statistics across adjudication results."""
    n = len(results)
    if n == 0:
        return {"count": 0}

    total_submitted = sum(r.total_submitted for r in results)
    total_paid = sum(r.total_paid for r in results)
    total_pr = sum(r.total_patient_responsibility for r in results)
    total_adjustments = sum(r.total_adjustments for r in results)

    n_fully_paid = sum(1 for r in results if r.claim_status == "paid")
    n_partial = sum(1 for r in results if r.claim_status == "partially_paid")
    n_denied = sum(1 for r in results if r.claim_status == "fully_denied")

    total_lines = sum(len(r.lines) for r in results)
    denied_lines = sum(
        sum(1 for l in r.lines if l.status == "denied")
        for r in results
    )
    bundled_lines = sum(
        sum(1 for l in r.lines if l.status == "bundled")
        for r in results
    )

    # Per-payer breakdown
    by_payer: dict[str, dict] = {}
    for r in results:
        pid = r.payer_id
        if pid not in by_payer:
            by_payer[pid] = {
                "payer_id": pid,
                "plan_type": r.payer_profile,
                "claims": 0,
                "submitted": 0.0,
                "paid": 0.0,
                "denied_lines": 0,
                "total_lines": 0,
            }
        by_payer[pid]["claims"] += 1
        by_payer[pid]["submitted"] += r.total_submitted
        by_payer[pid]["paid"] += r.total_paid
        by_payer[pid]["denied_lines"] += sum(
            1 for l in r.lines if l.status == "denied"
        )
        by_payer[pid]["total_lines"] += len(r.lines)

    for pid in by_payer:
        p = by_payer[pid]
        p["payment_rate"] = round(p["paid"] / max(p["submitted"], 0.01), 4)
        p["line_denial_rate"] = round(
            p["denied_lines"] / max(p["total_lines"], 1), 4
        )

    return {
        "count": n,
        "total_submitted": round(total_submitted, 2),
        "total_paid": round(total_paid, 2),
        "total_patient_responsibility": round(total_pr, 2),
        "total_adjustments": round(total_adjustments, 2),
        "payment_rate": round(total_paid / max(total_submitted, 0.01), 4),
        "claim_status": {
            "fully_paid": n_fully_paid,
            "partially_paid": n_partial,
            "fully_denied": n_denied,
        },
        "line_denial_rate": round(
            denied_lines / max(total_lines, 1), 4
        ),
        "line_bundle_rate": round(
            bundled_lines / max(total_lines, 1), 4
        ),
        "by_payer": list(by_payer.values()),
    }
