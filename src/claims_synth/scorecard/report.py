import json


def to_json(scorecard):
    return json.dumps(scorecard, indent=2)


def to_markdown(scorecard):
    lines = []
    lines.append("# claims-synth Fidelity Scorecard\n")
    lines.append(f"**Overall fidelity:** {scorecard['overall_fidelity']} / 100  (grade {scorecard['grade']})\n")
    if "validity_rate" in scorecard:
        lines.append(f"**Validity rate:** {scorecard['validity_rate']}\n")
    lines.append("| Dimension | Subscore | Metric |\n|---|---|---|\n")
    for name, dim in scorecard["dimensions"].items():
        lines.append(f"| {name} | {dim['subscore']} | {dim['metric']} |\n")
    lines.append("\n## Reference\n")
    provenance = scorecard.get("reference_provenance", {})
    for key, value in provenance.items():
        lines.append(f"- {key}: {value}\n")
    return "".join(lines)
