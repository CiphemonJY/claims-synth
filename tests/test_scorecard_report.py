"""Contract for claims_synth.scorecard.report — render a scorecard as JSON + markdown."""
import json
from claims_synth.scorecard.report import to_json, to_markdown


def _sc():
    return {
        "overall_fidelity": 87.5, "grade": "B", "validity_rate": 0.98,
        "dimensions": {"amount": {"subscore": 0.9, "metric": "1-KS"},
                       "los": {"subscore": 0.85, "metric": "1-JS"}},
        "reference_provenance": {"source": "CMS 2008-2010 DE-SynPUF", "n_claims": 66773},
    }


def test_to_json_roundtrips():
    d = json.loads(to_json(_sc()))
    assert d["overall_fidelity"] == 87.5 and d["grade"] == "B"


def test_to_markdown_contains_key_fields():
    md = to_markdown(_sc())
    assert isinstance(md, str)
    assert "87.5" in md and "B" in md
    assert "amount" in md and "los" in md
    assert "DE-SynPUF" in md            # provenance is rendered (real reference is cited)
