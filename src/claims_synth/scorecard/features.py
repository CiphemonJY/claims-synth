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


# ── Denial-risk table (ML-utility column) ──────────────────────────────────

#: Column that `denial_table` writes the binary outcome into.
DENIAL_TARGET = "denied"


def denial_table(claims: List[Claim], adj_seed: int = 42) -> dict:
    """Claim-level features + a binary denial label, as a dict of columns.

    The label is the roadmap's G5 outcome: 1 when the payer denied at least
    one line of the claim, 0 otherwise. Labels come from the rules-based
    adjudicator (G2), so the "outcome" is a deterministic function of the
    claim under a fixed payer ruleset — that is what makes TRTR/TSTR
    meaningful here: a generator that reproduces the joint structure the
    rules key off (payer x authorization x diagnosis/procedure compatibility
    x place of service) yields synthetic training rows a denial model can
    learn the same decision boundary from.

    Features are strictly pre-adjudication — nothing derived from the
    remittance is fed back in, or the model would be reading its own label.
    """
    from claims_synth.adjudicate import adjudicate_batch

    results = adjudicate_batch(claims, seed=adj_seed)
    cols = {
        "amount": [], "los": [], "n_dx": [], "n_proc": [],
        "n_modifiers": [], "units": [], "mean_line_charge": [],
        "dx_chapter": [], "claim_type": [], "plan_type": [],
        "place_of_service": [], "admission_type": [], "auth_status": [],
        DENIAL_TARGET: [],
    }
    for claim, res in zip(claims, results):
        enc = claim.encounter
        auth = claim.authorization
        n_lines = len(claim.lines)
        cols["amount"].append(float(claim.total_charge))
        cols["los"].append(claim_los(claim))
        cols["n_dx"].append(len(claim.diagnoses))
        cols["n_proc"].append(n_lines)
        cols["n_modifiers"].append(sum(len(l.modifiers) for l in claim.lines))
        cols["units"].append(sum(l.units for l in claim.lines))
        cols["mean_line_charge"].append(
            float(claim.total_charge) / n_lines if n_lines else 0.0)
        cols["dx_chapter"].append(
            icd10_chapter(claim.diagnoses[0].code) if claim.diagnoses else "Other")
        cols["claim_type"].append(str(getattr(claim.claim_type, "value", claim.claim_type)))
        cols["plan_type"].append(str(claim.payer.plan_type) if claim.payer else "Unknown")
        cols["place_of_service"].append(
            str(getattr(enc.place_of_service, "value", enc.place_of_service))
            if enc is not None else "Unknown")
        cols["admission_type"].append(
            str(enc.admission_type) if enc is not None and enc.admission_type else "None")
        cols["auth_status"].append(
            str(auth.auth_status) if auth is not None and auth.auth_status else "None")
        cols[DENIAL_TARGET].append(
            1 if any(l.status == "denied" for l in res.lines) else 0)
    return cols
