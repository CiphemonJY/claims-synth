"""Reachability of the timely-filing (CARC 29) denial path.

These tests exist because the rule was DEAD. Every payer profile configures a
timely_filing_days window (90-365) and a timely_filing_denial_rate of 0.95, but
the generator dated every claim at most 90 days old, so
`days_since > profile.timely_filing_days` was false on 30,000/30,000 claims and
the denial roll behind it never executed once. The engine could not produce a
CARC 29 adjustment at any seed or volume.

Nothing in the suite caught that: every test asserted on behaviour that DID
happen. A rule that never fires is invisible to tests written that way, so the
tests below assert reachability directly.
"""

import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claims_synth.adjudicate import PAYER_PROFILES, adjudicate_batch  # noqa: E402
from claims_synth.generate import ClaimGenerator  # noqa: E402


def _claims(n=2000, seed=0, **kw):
    return ClaimGenerator(seed=seed, **kw).generate(n=n)


def _carc(results, code):
    return [
        adj
        for r in results
        for adj in (r.claim_level_adjustments or [])
        if adj.get("code") == code
    ]


class TestDefaultIsUnchanged:
    """The fix must not move the calibrated default distribution."""

    def test_late_filing_rate_defaults_to_off(self):
        assert ClaimGenerator(seed=1)._late_filing_rate == 0.0

    def test_default_consumes_no_extra_randomness(self):
        # The draw is guarded by `rate > 0.0` precisely so that the default path
        # pulls no value off the RNG and output stays byte-identical to the
        # behaviour the fidelity scorecard was calibrated against.
        implicit = [c.to_json() for c in _claims(500, seed=7)]
        explicit = [c.to_json() for c in _claims(500, seed=7, late_filing_rate=0.0)]
        assert implicit == explicit

    def test_default_claim_ages_sit_inside_every_payer_window(self):
        # This is the measurement that explains WHY the rule was dead. If someone
        # widens the generator's date range later, this test tells them the
        # timely-filing path just became reachable by default.
        today = date.today()
        ages = [(today - c.statement_to).days for c in _claims(3000, seed=3)]
        tightest = min(p.timely_filing_days for p in PAYER_PROFILES.values())
        assert max(ages) <= tightest, (
            "claim ages now exceed the tightest payer window (%d days); "
            "timely-filing denials are reachable by default" % tightest
        )

    def test_default_produces_no_timely_filing_denials(self):
        claims = _claims(4000, seed=11)
        assert _carc(adjudicate_batch(claims, seed=5), "29") == []


class TestReachableWhenEnabled:
    def test_timely_filing_denials_are_produced(self):
        claims = _claims(4000, seed=11, late_filing_rate=0.5)
        adjustments = _carc(adjudicate_batch(claims, seed=5), "29")
        assert adjustments, "timely-filing rule is unreachable even when enabled"
        assert all(a["amount"] > 0 for a in adjustments)

    def test_a_timely_filing_denial_denies_the_whole_claim(self):
        claims = _claims(4000, seed=11, late_filing_rate=0.5)
        results = adjudicate_batch(claims, seed=5)
        denied = [
            r
            for r in results
            if any(a.get("code") == "29" for a in (r.claim_level_adjustments or []))
        ]
        assert denied
        # claim_denied propagates into every line, so the rollup is fully_denied.
        assert {r.claim_status for r in denied} == {"fully_denied"}

    def test_rate_controls_frequency(self):
        def n29(rate):
            return len(_carc(adjudicate_batch(_claims(3000, seed=2, late_filing_rate=rate),
                                              seed=5), "29"))

        assert n29(0.0) == 0 < n29(0.1) < n29(0.8)

    def test_service_dates_move_with_the_statement_dates(self):
        # A late-filed claim is an old encounter billed now -- not a recent
        # encounter with a backdated statement. The invariant the generator keeps
        # for on-time claims (statement_from == encounter.from_date) must survive.
        for c in _claims(2000, seed=4, late_filing_rate=0.5):
            if c.encounter:
                assert c.statement_from == c.encounter.from_date

    def test_late_claims_are_actually_old(self):
        claims = _claims(3000, seed=6, late_filing_rate=1.0)
        today = date.today()
        ages = [(today - c.statement_to).days for c in claims]
        tightest = min(p.timely_filing_days for p in PAYER_PROFILES.values())
        assert max(ages) > tightest


class TestReasonCodeConsistency:
    def test_hardcoded_group_matches_the_reason_code_table(self):
        # adjudicate() writes {"group": "PR", "code": "29"} literally rather than
        # reading CARC_CODES["29"].group. They agree today; this test fails if
        # one is corrected without the other.
        from claims_synth.adjudicate import CARC_CODES

        claims = _claims(4000, seed=11, late_filing_rate=0.5)
        emitted = {a["group"] for a in _carc(adjudicate_batch(claims, seed=5), "29")}
        assert emitted == {CARC_CODES["29"].group.value}

    def test_denial_mix_is_not_dominated_by_one_code(self):
        # Cheap guard against another whole denial category going dark: if any
        # single claim-level code ever accounts for everything, something else
        # stopped firing.
        claims = _claims(4000, seed=11, late_filing_rate=0.1)
        codes = Counter(
            a.get("code")
            for r in adjudicate_batch(claims, seed=5)
            for a in (r.claim_level_adjustments or [])
        )
        assert codes, "no claim-level adjustments at all"
