"""
Synthetic claim generator — deterministic by seed.

Produces institutional (837I) and professional (837P) claims from
configurable code/payer distributions. CMS DE-SynPUF-style mixes.
Zero copyrighted code descriptions — ICD-10 + HCPCS Level II only.
CPT treated as opaque user-supplied IDs via cpt.register_cpt_codes().
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from claims_synth.claim import (
    Claim, ClaimLine, ClaimType, Diagnosis, Encounter,
    Payer, Authorization, PlaceOfService,
)
from claims_synth.vocab.icd10 import ICD10_CODES, chronic_codes, cc_codes, mcc_codes
from claims_synth.vocab.hcpcs import (
    HCPCS_CODES, billable_codes, drug_codes, dme_codes,
    supply_codes, transport_codes, temp_codes, modifier_codes,
)
from claims_synth.vocab.cpt import all_codes as cpt_all_codes, register_cpt_codes


# ── Payer pool ─────────────────────────────────────────────────────────────
# Opaque IDs only — no real payer names shipped.

PAYER_POOL: list[dict] = [
    {"payer_id": "PAYER_A", "plan_type": "COMMERCIAL"},
    {"payer_id": "PAYER_B", "plan_type": "COMMERCIAL"},
    {"payer_id": "PAYER_C", "plan_type": "MEDICARE"},
    {"payer_id": "PAYER_D", "plan_type": "MEDICARE_ADVANTAGE"},
    {"payer_id": "PAYER_E", "plan_type": "MEDICAID"},
    {"payer_id": "PAYER_F", "plan_type": "COMMERCIAL"},
    {"payer_id": "PAYER_G", "plan_type": "MEDICARE_ADVANTAGE"},
    {"payer_id": "PAYER_H", "plan_type": "MEDICAID"},
]

PAYER_WEIGHTS = np.array([0.20, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10], dtype=np.float64)
PAYER_WEIGHTS /= PAYER_WEIGHTS.sum()


# ── Diagnosis frequency weights ────────────────────────────────────────────

def _build_dx_weights() -> np.ndarray:
    weights = np.ones(len(ICD10_CODES), dtype=np.float64)
    for i, c in enumerate(ICD10_CODES):
        if c.is_chronic:
            weights[i] *= 3.0
        if c.is_cc:
            weights[i] *= 1.5
        if c.is_mcc:
            weights[i] *= 0.5
        if c.chapter == "Symptoms":
            weights[i] *= 0.7
        if c.chapter == "Injury":
            weights[i] *= 0.4
        if c.chapter == "Pregnancy":
            weights[i] *= 0.3
    return weights / weights.sum()


DX_WEIGHTS = _build_dx_weights()


# ── Procedure frequency weights ────────────────────────────────────────────
# HCPCS Level II only. CPT codes are user-supplied and mixed in at runtime.

def _build_proc_weights() -> np.ndarray:
    billable = billable_codes()
    weights = np.ones(len(billable), dtype=np.float64)
    for i, c in enumerate(billable):
        if c.is_temp:
            weights[i] *= 3.0   # G-codes (visits, screenings) are common
        elif c.is_drug:
            weights[i] *= 1.5
        elif c.is_supply:
            weights[i] *= 1.2
        elif c.is_dme:
            weights[i] *= 0.3
        elif c.is_transport:
            weights[i] *= 0.2
    return weights / weights.sum()


PROC_WEIGHTS = _build_proc_weights()
_BILLABLE = billable_codes()
_MODIFIERS = modifier_codes()


# ── Charge ranges by group ─────────────────────────────────────────────────

CHARGE_RANGES: dict[str, tuple[float, float]] = {
    "Drugs":     (20.0, 5000.0),
    "DME":       (50.0, 1500.0),
    "Supplies":  (5.0, 200.0),
    "Transport": (300.0, 1500.0),
    "Temp":      (50.0, 500.0),
}


# ── Revenue code mapping ───────────────────────────────────────────────────

REVENUE_CODES: dict[str, str] = {
    "Drugs": "0250",
    "DME": "0270",
    "Supplies": "0270",
    "Transport": "0540",
    "Temp": "0510",
}


# ── Generator ──────────────────────────────────────────────────────────────

class ClaimGenerator:
    """
    Deterministic synthetic claim generator.

    All randomness is seeded — same seed produces identical output.
    Uses HCPCS Level II codes (public domain) by default.
    CPT codes can be mixed in via register_cpt_codes() before generation.
    """

    def __init__(self, seed: int = 42, cpt_codes: Optional[list[str]] = None,
                 cpt_weight: float = 0.0):
        """
        Args:
            seed: Random seed for reproducibility.
            cpt_codes: User-supplied CPT code list (opaque IDs).
            cpt_weight: Probability of selecting a CPT code vs HCPCS (0.0 = HCPCS only).
        """
        self.rng = np.random.RandomState(seed)
        self._claim_counter = 0
        self._cpt_weight = cpt_weight

        if cpt_codes:
            register_cpt_codes(cpt_codes, category="user-supplied")
        self._cpt_codes = cpt_all_codes()

        # Build combined procedure pool
        self._proc_pool: list[tuple[str, str, str, Optional[int]]] = []
        self._proc_weights: list[float] = []

        # HCPCS codes
        for c in _BILLABLE:
            lo, hi = CHARGE_RANGES.get(c.group, (50.0, 500.0))
            self._proc_pool.append((c.code, "HCPCS", c.group, c.mue_units))
            self._proc_weights.append(PROC_WEIGHTS[_BILLABLE.index(c)])

        # CPT codes (if supplied)
        if self._cpt_codes and cpt_weight > 0:
            cpt_count = len(self._cpt_codes)
            # Scale HCPCS weights down to make room for CPT
            hcpcs_total = sum(self._proc_weights)
            scale = (1.0 - cpt_weight) / hcpcs_total if hcpcs_total > 0 else 0
            self._proc_weights = [w * scale for w in self._proc_weights]
            # Add CPT codes with uniform weight
            cpt_w = cpt_weight / cpt_count
            for code in self._cpt_codes:
                self._proc_pool.append((code, "CPT", "CPT", None))
                self._proc_weights.append(cpt_w)

        self._proc_weights = np.array(self._proc_weights, dtype=np.float64)
        self._proc_weights /= self._proc_weights.sum()

    def generate(self, n: int = 1) -> list[Claim]:
        """Generate n claims."""
        claims = []
        for _ in range(n):
            self._claim_counter += 1
            claim_type = self._sample_claim_type()
            claim = self._build_claim(claim_type)
            claims.append(claim)
        return claims

    def _sample_claim_type(self) -> ClaimType:
        return ClaimType.PROFESSIONAL if self.rng.random() < 0.7 else ClaimType.INSTITUTIONAL

    def _build_claim(self, claim_type: ClaimType) -> Claim:
        # ── Diagnoses ──────────────────────────────────────────────────
        n_dx = self.rng.randint(1, 6)
        dx_indices = self.rng.choice(len(ICD10_CODES), size=n_dx, p=DX_WEIGHTS, replace=False)
        diagnoses = []
        for idx in dx_indices:
            dx_code = ICD10_CODES[idx]
            poa = None
            if claim_type == ClaimType.INSTITUTIONAL:
                poa = self.rng.choice([True, False, None], p=[0.85, 0.10, 0.05])
            diagnoses.append(Diagnosis(code=dx_code.code, present_on_admission=poa))

        # ── Encounter ──────────────────────────────────────────────────
        encounter = self._build_encounter(claim_type)

        # ── Lines ───────────────────────────────────────────────────────
        n_lines = self.rng.randint(1, 6)
        lines = []
        for seq in range(1, n_lines + 1):
            line = self._build_line(seq, diagnoses, claim_type)
            lines.append(line)

        # ── Payer ───────────────────────────────────────────────────────
        payer_idx = self.rng.choice(len(PAYER_POOL), p=PAYER_WEIGHTS)
        payer_data = PAYER_POOL[payer_idx]
        payer = Payer(payer_id=payer_data["payer_id"], plan_type=payer_data["plan_type"])

        # ── Authorization ──────────────────────────────────────────────
        auth = self._build_authorization()

        # ── Statement dates ────────────────────────────────────────────
        if encounter:
            stmt_from = encounter.from_date
            stmt_to = encounter.to_date or encounter.from_date
        else:
            stmt_from = date.today() - timedelta(days=self.rng.randint(0, 30))
            stmt_to = stmt_from

        claim = Claim(
            claim_id=f"C{self._claim_counter:08d}",
            claim_type=claim_type,
            patient_id=f"PT{self.rng.randint(10000, 99999):05d}",
            diagnoses=diagnoses,
            lines=lines,
            encounter=encounter,
            payer=payer,
            authorization=auth,
            statement_from=stmt_from,
            statement_to=stmt_to,
        )

        errors = claim.validate()
        if errors:
            raise RuntimeError(f"Generated invalid claim {claim.claim_id}: {errors}")

        return claim

    def _build_encounter(self, claim_type: ClaimType) -> Optional[Encounter]:
        if claim_type == ClaimType.PROFESSIONAL and self.rng.random() < 0.3:
            return None

        days_ago = self.rng.randint(0, 90)
        from_date = date.today() - timedelta(days=days_ago)

        if claim_type == ClaimType.INSTITUTIONAL:
            los = self.rng.randint(1, 14)
            to_date = from_date + timedelta(days=los)
        else:
            to_date = None

        if claim_type == ClaimType.INSTITUTIONAL:
            pos_weights = [0.05, 0.50, 0.25, 0.15, 0.03, 0.02]
        else:
            pos_weights = [0.60, 0.0, 0.20, 0.10, 0.05, 0.05]
        pos_names = ["OFFICE", "INPATIENT", "OUTPATIENT", "EMERGENCY", "ASC", "TELEHEALTH"]
        pos_name = self.rng.choice(pos_names, p=np.array(pos_weights) / sum(pos_weights))
        pos = PlaceOfService[pos_name]

        admission_type = None
        discharge_status = None
        if claim_type == ClaimType.INSTITUTIONAL:
            admission_type = self.rng.choice(["1", "2", "3"], p=[0.3, 0.2, 0.5])
            discharge_status = self.rng.choice(
                ["01", "02", "03", "04", "06", "07", "20", "30"],
                p=[0.5, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.1],
            )

        return Encounter(
            from_date=from_date,
            to_date=to_date,
            place_of_service=pos,
            admission_type=admission_type,
            discharge_status=discharge_status,
        )

    def _build_line(self, seq: int, diagnoses: list[Diagnosis], claim_type: ClaimType) -> ClaimLine:
        # Sample procedure from combined pool
        idx = self.rng.choice(len(self._proc_pool), p=self._proc_weights)
        code, code_type, group, mue = self._proc_pool[idx]

        # Modifiers
        n_mods = self.rng.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
        mod_indices = self.rng.choice(len(_MODIFIERS), size=n_mods, replace=False) if n_mods > 0 else []
        modifiers = [_MODIFIERS[i].code for i in mod_indices]

        # Diagnosis pointers
        n_ptrs = min(self.rng.randint(1, 5), len(diagnoses))
        ptrs = sorted(self.rng.choice(len(diagnoses), size=n_ptrs, replace=False) + 1)

        # Charge
        lo, hi = CHARGE_RANGES.get(group, (50.0, 500.0))
        charge = round(self.rng.uniform(lo, hi), 2)

        # Units
        if mue:
            units = self.rng.randint(1, min(mue, 4) + 1)
        else:
            units = 1

        # Revenue code (institutional only)
        revenue_code = None
        if claim_type == ClaimType.INSTITUTIONAL:
            revenue_code = REVENUE_CODES.get(group, "0510")

        return ClaimLine(
            sequence=seq,
            procedure_code=code,
            procedure_code_type=code_type,
            modifiers=modifiers,
            diagnosis_pointers=[int(p) for p in ptrs],
            charge=charge,
            units=units,
            revenue_code=revenue_code,
        )

    def _build_authorization(self) -> Optional[Authorization]:
        if self.rng.random() > 0.4:
            return None

        auth_number = f"AUTH{self.rng.randint(100000, 999999):06d}" if self.rng.random() < 0.7 else None
        auth_status = self.rng.choice(["approved", "denied", "not_required"], p=[0.6, 0.15, 0.25]) if auth_number else None
        referral_number = f"REF{self.rng.randint(10000, 99999):05d}" if self.rng.random() < 0.2 else None

        return Authorization(
            auth_number=auth_number,
            auth_status=auth_status,
            referral_number=referral_number,
        )


# ── CLI ────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="claims-synth: generate synthetic healthcare claims"
    )
    parser.add_argument("--n", type=int, default=100, help="Number of claims to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (deterministic)")
    parser.add_argument("--output", type=str, default=None, help="Output file (JSONL). stdout if omitted.")
    parser.add_argument("--type", type=str, choices=["837I", "837P", "mixed"], default="mixed",
                        help="Claim type filter")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON (one per line)")
    parser.add_argument("--cpt-codes", type=str, default=None,
                        help="Comma-separated user-supplied CPT codes (opaque IDs)")
    parser.add_argument("--cpt-weight", type=float, default=0.0,
                        help="Probability of selecting CPT vs HCPCS (0.0-1.0)")
    args = parser.parse_args(argv)

    cpt_codes = None
    if args.cpt_codes:
        cpt_codes = [c.strip() for c in args.cpt_codes.split(",") if c.strip()]

    gen = ClaimGenerator(seed=args.seed, cpt_codes=cpt_codes, cpt_weight=args.cpt_weight)
    claims = gen.generate(n=args.n)

    if args.type == "837I":
        claims = [c for c in claims if c.claim_type == ClaimType.INSTITUTIONAL]
    elif args.type == "837P":
        claims = [c for c in claims if c.claim_type == ClaimType.PROFESSIONAL]

    out_fh = open(args.output, "w") if args.output else sys.stdout
    try:
        for claim in claims:
            indent = 2 if args.pretty else None
            line = claim.to_json(indent=indent)
            if args.pretty:
                out_fh.write(line + "\n")
            else:
                out_fh.write(line.replace("\n", "") + "\n")
    finally:
        if args.output:
            out_fh.close()

    if args.output:
        print(f"Generated {len(claims)} claims → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
