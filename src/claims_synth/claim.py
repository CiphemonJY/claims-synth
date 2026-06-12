"""
Canonical claim object for claims-synth.

Represents both institutional (837I) and professional (837P) claims.
Zero copyrighted code descriptions — CPT codes are opaque user-supplied IDs.
ICD-10 and HCPCS Level II are public domain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date
from enum import Enum
from typing import Optional
from uuid import uuid4


class ClaimType(str, Enum):
    INSTITUTIONAL = "837I"
    PROFESSIONAL = "837P"


class PlaceOfService(str, Enum):
    """CMS Place of Service codes (public domain subset)."""
    OFFICE = "11"
    INPATIENT = "21"
    OUTPATIENT = "22"
    EMERGENCY = "23"
    ASC = "24"
    TELEHEALTH = "02"


@dataclass
class Diagnosis:
    """
    An ICD-10-CM diagnosis code.

    ICD-10 codes are public domain (WHO + CMS publish them freely).
    We ship a curated subset — no copyrighted descriptions.
    """
    code: str                          # e.g. "I10", "E11.9"
    present_on_admission: Optional[bool] = None  # POA indicator (institutional)


@dataclass
class ClaimLine:
    """
    A single service line on a claim.

    CPT codes are treated as opaque user-supplied IDs.
    HCPCS Level II codes are public domain.
    """
    sequence: int                      # line number (1-indexed)
    procedure_code: str                # HCPCS Level II or opaque CPT ID
    procedure_code_type: str = "HCPCS"  # "HCPCS" or "CPT"
    modifiers: list[str] = field(default_factory=list)  # up to 4
    diagnosis_pointers: list[int] = field(default_factory=list)  # 1-indexed refs to claim.diagnoses
    charge: float = 0.0               # submitted charge in dollars
    units: int = 1                     # service units
    revenue_code: Optional[str] = None  # institutional only (e.g. "0450")
    ndc: Optional[str] = None          # NDC code if drug line


@dataclass
class Encounter:
    """Encounter/visit information."""
    from_date: date
    to_date: Optional[date] = None     # None = same-day
    place_of_service: PlaceOfService = PlaceOfService.OFFICE
    admission_type: Optional[str] = None  # institutional: "1"=emergency, "2"=urgent, "3"=elective
    discharge_status: Optional[str] = None  # institutional: "01"=home, etc.


@dataclass
class Payer:
    """Payer information — opaque ID, no real payer names shipped."""
    payer_id: str                      # opaque identifier (e.g. "PAYER_A")
    plan_type: Optional[str] = None     # "COMMERCIAL", "MEDICARE", "MEDICAID", "MEDICARE_ADVANTAGE"


@dataclass
class Authorization:
    """Prior authorization / referral fields."""
    auth_number: Optional[str] = None
    auth_status: Optional[str] = None  # "approved", "denied", "not_required"
    referral_number: Optional[str] = None


@dataclass
class Claim:
    """
    Canonical claim object.

    Represents one complete claim — institutional or professional.
    Deterministic by seed when generated through claims_synth.generate.
    """
    claim_id: str = field(default_factory=lambda: uuid4().hex[:12])
    claim_type: ClaimType = ClaimType.PROFESSIONAL
    patient_id: str = ""               # de-identified
    diagnoses: list[Diagnosis] = field(default_factory=list)  # primary first
    lines: list[ClaimLine] = field(default_factory=list)
    encounter: Optional[Encounter] = None
    payer: Optional[Payer] = None
    authorization: Optional[Authorization] = None
    total_charge: float = 0.0          # sum of line charges
    statement_from: Optional[date] = None
    statement_to: Optional[date] = None

    def __post_init__(self):
        if self.total_charge == 0.0 and self.lines:
            self.total_charge = sum(line.charge * line.units for line in self.lines)

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors: list[str] = []

        if not self.claim_id:
            errors.append("claim_id is required")

        if not self.lines:
            errors.append("at least one claim line is required")

        for i, line in enumerate(self.lines):
            if line.sequence != i + 1:
                errors.append(f"line {i} sequence mismatch: expected {i+1}, got {line.sequence}")
            if not line.procedure_code:
                errors.append(f"line {line.sequence}: procedure_code is required")
            if len(line.modifiers) > 4:
                errors.append(f"line {line.sequence}: max 4 modifiers, got {len(line.modifiers)}")
            if line.charge < 0:
                errors.append(f"line {line.sequence}: charge must be non-negative")
            if line.units < 1:
                errors.append(f"line {line.sequence}: units must be >= 1")
            for ptr in line.diagnosis_pointers:
                if ptr < 1 or ptr > len(self.diagnoses):
                    errors.append(f"line {line.sequence}: diagnosis pointer {ptr} out of range (1-{len(self.diagnoses)})")

        if not self.diagnoses:
            errors.append("at least one diagnosis is required")

        if self.claim_type == ClaimType.INSTITUTIONAL:
            if not self.encounter:
                errors.append("institutional claim requires encounter")
            for line in self.lines:
                if not line.revenue_code:
                    errors.append(f"line {line.sequence}: institutional claim requires revenue_code")

        if self.encounter and self.encounter.to_date and self.encounter.from_date > self.encounter.to_date:
            errors.append("encounter from_date after to_date")

        if self.statement_from and self.statement_to and self.statement_from > self.statement_to:
            errors.append("statement_from after statement_to")

        computed = sum(line.charge * line.units for line in self.lines)
        if abs(computed - self.total_charge) > 0.01:
            errors.append(f"total_charge mismatch: computed {computed:.2f}, stored {self.total_charge:.2f}")

        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        d = asdict(self)
        d["claim_type"] = self.claim_type.value
        if self.encounter:
            d["encounter"]["from_date"] = self.encounter.from_date.isoformat()
            if self.encounter.to_date:
                d["encounter"]["to_date"] = self.encounter.to_date.isoformat()
            d["encounter"]["place_of_service"] = self.encounter.place_of_service.value
        if self.statement_from:
            d["statement_from"] = self.statement_from.isoformat()
        if self.statement_to:
            d["statement_to"] = self.statement_to.isoformat()
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> Claim:
        """Deserialize from dict."""
        d = dict(d)  # shallow copy
        d["claim_type"] = ClaimType(d["claim_type"])
        d["diagnoses"] = [Diagnosis(**dx) for dx in d.get("diagnoses", [])]
        d["lines"] = [ClaimLine(**line) for line in d.get("lines", [])]
        if d.get("encounter"):
            enc = d["encounter"]
            enc["from_date"] = date.fromisoformat(enc["from_date"])
            if enc.get("to_date"):
                enc["to_date"] = date.fromisoformat(enc["to_date"])
            enc["place_of_service"] = PlaceOfService(enc["place_of_service"])
            d["encounter"] = Encounter(**enc)
        if d.get("payer"):
            d["payer"] = Payer(**d["payer"])
        if d.get("authorization"):
            d["authorization"] = Authorization(**d["authorization"])
        if d.get("statement_from"):
            d["statement_from"] = date.fromisoformat(d["statement_from"])
        if d.get("statement_to"):
            d["statement_to"] = date.fromisoformat(d["statement_to"])
        return cls(**d)
