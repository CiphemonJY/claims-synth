import json


def to_json(scorecard):
    return json.dumps(scorecard, indent=2)


def _band(entry, key):
    """'mean ± sd' for a metric written by ml_utility's flattened bands."""
    if entry is None or entry.get(key) is None:
        return "—"
    sd = entry.get(key + "_sd")
    return f"{entry[key]:.4f}" if sd is None else f"{entry[key]:.4f} ± {sd:.4f}"


def _ml_utility_section(ml):
    lines = ["\n## ML utility — TRTR vs TSTR\n"]
    if "skipped" in ml:
        lines.append(f"\n_Not scored: {ml['skipped']}._\n")
        return lines
    lines.append(f"\n- Target: {ml.get('target', 'binary outcome')}\n")
    lines.append(f"- Held-out population: {ml.get('real_population', 'caller-supplied')}\n")
    lines.append(f"- Candidate generator: {ml.get('candidate_generator', 'caller-supplied')}\n")
    lines.append(f"- Predictor: {ml.get('predictor', 'custom')}; "
                 f"{ml.get('n_repeats', 1)} split-and-fit repeats; "
                 f"train n={ml['trtr'].get('n_train')} per arm, test n={ml.get('n_test')}\n")
    lines.append(f"- ECE over {ml.get('ece_bins')} {ml.get('ece_binning', '')} bins. "
                 "AUC is invariant to any monotone rescaling of the scores, so it cannot see "
                 "miscalibration; the denial-risk model downstream consumes a probability, "
                 "not a ranking.\n")
    lines.append("\n| Arm | AUC ↑ | Brier ↓ | ECE ↓ |\n|---|---|---|---|\n")
    rows = [("TRTR (real-trained)", ml["trtr"]),
            ("TSTR (synthetic-trained)", ml["tstr"])]
    for label, arm in list(rows):
        if "calibrated" in arm:
            rows.append((label.split(" (")[0] + " + isotonic", arm["calibrated"]))
    for label, entry in rows:
        lines.append(f"| {label} | {_band(entry, 'auc')} | {_band(entry, 'brier')} "
                     f"| {_band(entry, 'ece')} |\n")
    lo, hi = ml.get("utility_ratio_min"), ml.get("utility_ratio_max")
    span = f"  (range {lo:.4f}–{hi:.4f})" if lo is not None else ""
    lines.append(f"\n**Utility ratio** (chance-corrected, TSTR vs TRTR): "
                 f"{_band(ml, 'utility_ratio')}{span}\n")
    if ml.get("resolution"):
        lines.append(f"\n_Resolution — {ml['resolution']}._\n")
    return lines


def to_markdown(scorecard):
    lines = []
    lines.append("# claims-synth Fidelity Scorecard\n")
    lines.append(f"**Overall fidelity:** {scorecard['overall_fidelity']} / 100  (grade {scorecard['grade']})\n")
    if "validity_rate" in scorecard:
        lines.append(f"**Validity rate:** {scorecard['validity_rate']}\n")
    lines.append("| Dimension | Subscore | Metric |\n|---|---|---|\n")
    for name, dim in scorecard["dimensions"].items():
        lines.append(f"| {name} | {dim['subscore']} | {dim['metric']} |\n")
    if scorecard.get("ml_utility"):
        lines.extend(_ml_utility_section(scorecard["ml_utility"]))
    lines.append("\n## Reference\n")
    provenance = scorecard.get("reference_provenance", {})
    for key, value in provenance.items():
        lines.append(f"- {key}: {value}\n")
    return "".join(lines)
