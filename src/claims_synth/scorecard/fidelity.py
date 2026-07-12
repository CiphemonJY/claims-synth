from claims_synth.scorecard.stats import ks, js, to_dist

def score(features, reference, validity_rate=None):
    dims = {}
    # continuous: amount
    if "amount" in reference and reference["amount"].get("kind")=="continuous":
        sub = 1.0 - ks(reference["amount"]["sorted"], features.get("amount", []))
        dims["amount"] = {"subscore": round(sub,4), "metric":"1-KS"}
    # discrete: los, n_dx, n_proc
    for dim in ("los","n_dx","n_proc"):
        if dim in reference and reference[dim].get("dist") is not None:
            cand = to_dist(features.get(dim, []))
            ref_dist = reference[dim]["dist"]
            keys = sorted(set(cand) | set(ref_dist))
            sub = 1.0 - js(cand, ref_dist, keys)
            dims[dim] = {"subscore": round(sub,4), "metric":"1-JS"}
    # categorical: dx_chapter (keys come from the reference taxonomy so both distributions align)
    if "dx_chapter" in reference and reference["dx_chapter"].get("dist") is not None:
        cand = to_dist(features.get("dx_chapter", []))
        ref_dist = reference["dx_chapter"]["dist"]
        keys = reference.get("dx_chapters") or sorted(set(cand) | set(ref_dist))
        sub = 1.0 - js(cand, ref_dist, keys)
        dims["dx_chapter"] = {"subscore": round(sub,4), "metric":"1-JS"}
    overall = round(100.0 * (sum(d["subscore"] for d in dims.values())/len(dims)), 2) if dims else 0.0
    grade = "A" if overall>=90 else "B" if overall>=80 else "C" if overall>=70 else "D" if overall>=60 else "F"
    result = {"overall_fidelity": overall, "grade": grade, "dimensions": dims,
              "reference_provenance": reference.get("provenance", {})}
    if validity_rate is not None:
        result["validity_rate"] = validity_rate
    return result
