from datetime import date
from typing import List, Optional

from claims_synth.claim import Claim, ClaimType
from claims_synth.scorecard.taxonomy import icd10_chapter


def claim_los(claim: Claim) -> int:
    enc = claim.encounter
    if enc is None or getattr(enc, 'to_date', None) is None:
        return 0
    d = (enc.to_date - enc.from_date).days
    return d if d > 0 else 0


def filter_claims(claims: List[Claim], claim_type: str = "all") -> List[Claim]:
    """Filter claims by type before scoring.

    'institutional' (837I) matches the DE-SynPUF inpatient reference for an apples-to-apples
    comparison of amount/LOS; 'professional' keeps 837P; 'all' keeps everything.
    """
    ct = (claim_type or "all").lower()
    if ct == "all":
        return list(claims)
    if ct == "institutional":
        return [c for c in claims if c.claim_type == ClaimType.INSTITUTIONAL]
    if ct == "professional":
        return [c for c in claims if c.claim_type == ClaimType.PROFESSIONAL]
    raise ValueError(
        f"unknown claim_type {claim_type!r} (expected 'all', 'institutional', or 'professional')"
    )


def dataset_features(claims: List[Claim]) -> dict:
    return {
        "amount": [c.total_charge for c in claims],
        "los": [claim_los(c) for c in claims],
        "n_dx": [len(c.diagnoses) for c in claims],
        "n_proc": [len(c.lines) for c in claims],
        "dx_chapter": [icd10_chapter(c.diagnoses[0].code) if c.diagnoses else "Other" for c in claims],
    }


def validity_rate(claims: List[Claim]) -> float:
    if not claims:
        return 0.0
    return sum(1 for c in claims if c.validate() == []) / len(claims)
