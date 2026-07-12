from collections import Counter
import math
import bisect

def ks(ref_sorted, values):
    ref = sorted(list(ref_sorted))
    vals = sorted(list(values))
    if len(ref) == 0 or len(vals) == 0:
        return 1.0
    n_ref = len(ref)
    n_vals = len(vals)
    union = sorted(set(ref + vals))
    max_diff = 0.0
    for x in union:
        ecdf_ref = bisect.bisect_right(ref, x) / n_ref
        ecdf_val = bisect.bisect_right(vals, x) / n_vals
        diff = abs(ecdf_ref - ecdf_val)
        if diff > max_diff:
            max_diff = diff
    return max_diff

def js(dist_a, dist_b, keys):
    if not keys:
        return 0.0
    p = [dist_a.get(k, 0) for k in keys]
    q = [dist_b.get(k, 0) for k in keys]
    sum_p = sum(p)
    sum_q = sum(q)
    if sum_p == 0:
        p = [1.0 / len(keys)] * len(keys)
    else:
        p = [v / sum_p for v in p]
    if sum_q == 0:
        q = [1.0 / len(keys)] * len(keys)
    else:
        q = [v / sum_q for v in q]
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    jsd = 0.0
    for pi, mi in zip(p, m):
        if pi > 0 and mi > 0:
            jsd += pi * math.log2(pi / mi)
    for qi, mi in zip(q, m):
        if qi > 0 and mi > 0:
            jsd += qi * math.log2(qi / mi)
    jsd *= 0.5
    return max(0.0, min(1.0, jsd))

def to_dist(values):
    c = Counter(str(v) for v in values)
    n = sum(c.values())
    if n == 0:
        return {}
    return {k: v / n for k, v in c.items()}
