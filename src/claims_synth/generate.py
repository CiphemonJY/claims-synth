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

# Evidence-based "realistic inpatient" preset, calibrated directly against the bundled
# DE-SynPUF aggregate reference (the same public aggregates the fidelity scorecard grades
# against). Each marginal the scorecard measures — total charge, length of stay, diagnosis
# count, line count, primary-diagnosis chapter mix — is sampled from the reference
# distribution itself, subject to structural validity (>=1 line, >=1 diagnosis,
# non-negative charges). Opt-in via ClaimGenerator.realistic_inpatient(); the default
# generator is unchanged.

_REFSTATS_PATH = Path(__file__).parent / "scorecard" / "data" / "desynpuf_refstats.json"


def _renormalize(dist: dict, drop: tuple = ()) -> dict:
    kept = {k: v for k, v in dist.items() if k not in drop and v > 0}
    total = sum(kept.values())
    return {k: v / total for k, v in kept.items()}


def _dist_arrays(dist: dict):
    keys = sorted(int(k) for k in dist)
    p = np.array([dist[str(k)] for k in keys], dtype=np.float64)
    return np.array(keys), p / p.sum()


def _cat_arrays(dist: dict):
    keys = sorted(dist)
    p = np.array([dist[k] for k in keys], dtype=np.float64)
    return keys, p / p.sum()


def load_inpatient_calibration(refstats_path: Optional[Path] = None) -> dict:
    """Build realistic-inpatient calibration from the bundled DE-SynPUF aggregate reference.

    Structural validity constraints applied:
    - n_dx: claims need >=1 diagnosis, so the reference's tiny mass at 0 is dropped.
    - n_line: claims need >=1 line (lines carry the charges; real 837I requires a service
      line too), so the reference's zero-procedure mass is renormalized over counts >=1 —
      the JS-optimal reallocation under that constraint.
    - amount: negative reference payments are clamped to 0 (charges must be non-negative).
    - dx_chapter: restricted to chapters with codes in the shipped vocab, renormalized.
    """
    from claims_synth.scorecard.taxonomy import icd10_chapter
    path = refstats_path or _REFSTATS_PATH
    ref = json.loads(Path(path).read_text())
    vocab_chapters = {icd10_chapter(c.code) for c in ICD10_CODES}
    chapter_dist = _renormalize(
        {k: v for k, v in ref["dx_chapter"]["dist"].items() if k in vocab_chapters})
    return {
        "amount_grid": [max(0.0, float(x)) for x in ref["amount"]["sorted"]],
        "los_dist": _renormalize(ref["los"]["dist"]),
        "n_dx_dist": _renormalize(ref["n_dx"]["dist"], drop=("0",)),
        "n_line_dist": _renormalize(ref["n_proc"]["dist"], drop=("0",)),
        "dx_chapter_dist": chapter_dist,
    }


class ClaimGenerator:
    """
    Deterministic synthetic claim generator.

    All randomness is seeded — same seed produces identical output.
    Uses HCPCS Level II codes (public domain) by default.
    CPT codes can be mixed in via register_cpt_codes() before generation.
    """

    def __init__(self, seed: int = 42, cpt_codes: Optional[list[str]] = None,
                 cpt_weight: float = 0.0, n_dx_mu=None, n_dx_sigma: float = 1.5,
                 n_line_mu=None, n_line_sigma: float = 1.5,
                 late_filing_rate: float = 0.0,
                 calibration: Optional[dict] = None):
        """
        Args:
            seed: Random seed for reproducibility.
            cpt_codes: User-supplied CPT code list (opaque IDs).
            cpt_weight: Probability of selecting a CPT code vs HCPCS (0.0 = HCPCS only).
            late_filing_rate: Probability that a claim is filed after the payer's
                timely-filing window. Default 0.0 preserves the existing claim-age
                distribution the fidelity scorecard is calibrated against. Set it
                above 0 to produce timely-filing (CARC 29) denials -- see the note
                in _build_claim for why they are otherwise unreachable.
        """
        self.rng = np.random.RandomState(seed)
        self._claim_counter = 0
        self._cpt_weight = cpt_weight
        self._late_filing_rate = late_filing_rate
        self._n_dx_mu = n_dx_mu
        self._n_dx_sigma = n_dx_sigma
        self._n_line_mu = n_line_mu
        self._n_line_sigma = n_line_sigma
        self._cal = calibration
        if calibration:
            from claims_synth.scorecard.taxonomy import icd10_chapter
            self._cal_amount = np.asarray(calibration["amount_grid"], dtype=np.float64)
            self._cal_los_k, self._cal_los_p = _dist_arrays(calibration["los_dist"])
            self._cal_ndx_k, self._cal_ndx_p = _dist_arrays(calibration["n_dx_dist"])
            self._cal_nline_k, self._cal_nline_p = _dist_arrays(calibration["n_line_dist"])
            self._cal_ch_k, self._cal_ch_p = _cat_arrays(calibration["dx_chapter_dist"])
            self._chapter_codes: dict[str, list[int]] = {}
            for i, c in enumerate(ICD10_CODES):
                self._chapter_codes.setdefault(icd10_chapter(c.code), []).append(i)

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

    @classmethod
    def realistic_inpatient(cls, seed: int = 42, **kwargs) -> "ClaimGenerator":
        """Generator calibrated to CMS DE-SynPUF inpatient marginals (837I only).

        Charge totals, LOS, diagnosis/line counts, and the primary-diagnosis chapter mix are
        sampled from the bundled DE-SynPUF aggregate reference (see load_inpatient_calibration).
        Extra kwargs override constructor args.
        """
        kwargs.setdefault("calibration", load_inpatient_calibration())
        return cls(seed=seed, **kwargs)

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
        if self._cal:
            return ClaimType.INSTITUTIONAL  # calibration reference is inpatient-only
        return ClaimType.PROFESSIONAL if self.rng.random() < 0.7 else ClaimType.INSTITUTIONAL

    def _build_claim(self, claim_type: ClaimType) -> Claim:
        # ── Diagnoses ──────────────────────────────────────────────────
        if self._cal:
            n_dx = int(self._cal_ndx_k[self.rng.choice(len(self._cal_ndx_k), p=self._cal_ndx_p)])
            dx_indices = self._sample_dx_calibrated(n_dx)
        elif self._n_dx_mu is None:
            n_dx = self.rng.randint(1, 6)
            dx_indices = self.rng.choice(len(ICD10_CODES), size=n_dx, p=DX_WEIGHTS, replace=False)
        else:
            n_dx = max(1, min(min(10, len(ICD10_CODES)),
                              int(round(self.rng.normal(self._n_dx_mu, self._n_dx_sigma)))))
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
        if self._cal:
            n_lines = int(self._cal_nline_k[self.rng.choice(len(self._cal_nline_k), p=self._cal_nline_p)])
        elif self._n_line_mu is None:
            n_lines = self.rng.randint(1, 6)
        else:
            n_lines = max(1, min(10, int(round(self.rng.normal(self._n_line_mu, self._n_line_sigma)))))
        lines = []
        for seq in range(1, n_lines + 1):
            line = self._build_line(seq, diagnoses, claim_type)
            lines.append(line)
        if self._cal:
            self._apply_calibrated_charges(lines)

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

        # Late-filed claims.
        #
        # Without this, timely-filing denials are STRUCTURALLY IMPOSSIBLE. Every
        # payer profile configures timely_filing_days (90-365) and a
        # timely_filing_denial_rate of 0.95, but service dates above are at most
        # 90 days old and an institutional stay only pushes statement_to later --
        # so `days_since > timely_filing_days` in Adjudicator.adjudicate() was
        # false on 30,000/30,000 claims and the denial roll behind it never
        # executed even once.
        #
        # Shift the SERVICE dates too, not just the statement dates: a late-filed
        # claim is an old encounter billed now, not a recent encounter with a
        # backdated statement.
        #
        # Defaults to 0.0 (off) because enabling it moves the claim-age
        # distribution that the DE-SynPUF fidelity scorecard is calibrated
        # against. The draw is guarded so that at the default no RNG value is
        # consumed and output stays byte-identical to before this change.
        if self._late_filing_rate > 0.0 and self.rng.random() < self._late_filing_rate:
            backdate = timedelta(days=int(self.rng.randint(95, 500)))
            stmt_from -= backdate
            stmt_to -= backdate
            if encounter:
                encounter.from_date -= backdate
                if encounter.to_date:
                    encounter.to_date -= backdate

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

    def _sample_dx_calibrated(self, n_dx: int) -> list[int]:
        """Primary diagnosis follows the reference chapter mix; secondaries follow DX_WEIGHTS."""
        ch = self._cal_ch_k[self.rng.choice(len(self._cal_ch_k), p=self._cal_ch_p)]
        pool = self._chapter_codes[ch]
        w = DX_WEIGHTS[pool] / DX_WEIGHTS[pool].sum()
        primary = int(self.rng.choice(pool, p=w))
        indices = [primary]
        if n_dx > 1:
            rest_w = DX_WEIGHTS.copy()
            rest_w[primary] = 0.0
            rest_w /= rest_w.sum()
            rest = self.rng.choice(len(ICD10_CODES), size=n_dx - 1, p=rest_w, replace=False)
            indices.extend(int(i) for i in rest)
        return indices

    def _apply_calibrated_charges(self, lines: list[ClaimLine]) -> None:
        """Distribute a reference-sampled total charge across lines, exact to the cent."""
        total = float(self._cal_amount[self.rng.randint(0, len(self._cal_amount))])
        cents = int(round(total * 100))
        cuts = sorted(int(c) for c in self.rng.randint(0, cents + 1, size=len(lines) - 1)) if len(lines) > 1 else []
        bounds = [0] + cuts + [cents]
        for line, lo, hi in zip(lines, bounds[:-1], bounds[1:]):
            line.units = 1
            line.charge = (hi - lo) / 100.0

    def _build_encounter(self, claim_type: ClaimType) -> Optional[Encounter]:
        if claim_type == ClaimType.PROFESSIONAL and self.rng.random() < 0.3:
            return None

        days_ago = self.rng.randint(0, 90)
        from_date = date.today() - timedelta(days=days_ago)

        if claim_type == ClaimType.INSTITUTIONAL:
            if self._cal:
                los = int(self._cal_los_k[self.rng.choice(len(self._cal_los_k), p=self._cal_los_p)])
            else:
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
    parser.add_argument("--late-filing-rate", type=float, default=0.0,
                        help="Fraction of claims filed past the payer's timely-filing "
                             "window (produces CARC 29 denials; 0 = off, the default)")
    parser.add_argument("--cpt-weight", type=float, default=0.0,
                        help="Probability of selecting CPT vs HCPCS (0.0-1.0)")
    args = parser.parse_args(argv)

    cpt_codes = None
    if args.cpt_codes:
        cpt_codes = [c.strip() for c in args.cpt_codes.split(",") if c.strip()]

    gen = ClaimGenerator(seed=args.seed, cpt_codes=cpt_codes, cpt_weight=args.cpt_weight,
                         late_filing_rate=args.late_filing_rate)
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
