from collections import Counter
import datetime
import json
import statistics

from claims_synth.scorecard.taxonomy import DX_CHAPTERS, icd9_chapter


def _num(s):
    try:
        return float(str(s).strip())
    except Exception:
        return None


def _norm_counter(counter):
    n = sum(counter.values())
    return {} if n == 0 else {str(k): v / n for k, v in counter.items()}


def build_reference(inpatient_rows, beneficiary_rows, provenance, downsample=1000):
    amounts = []
    los_list = []
    ndx_list = []
    nproc_list = []
    chap_counter = Counter()

    for row in inpatient_rows:
        a = _num(row.get("CLM_PMT_AMT"))
        if a is not None:
            amounts.append(a)

        lo = _num(row.get("CLM_UTLZTN_DAY_CNT"))
        if lo is not None:
            los = int(lo)
        else:
            adm_str = row.get("CLM_ADMSN_DT", "") or ""
            dis_str = row.get("NCH_BENE_DSCHRG_DT", "") or ""
            try:
                adm = datetime.date(int(adm_str[:4]), int(adm_str[4:6]), int(adm_str[6:8]))
                dis = datetime.date(int(dis_str[:4]), int(dis_str[4:6]), int(dis_str[6:8]))
                los = (dis - adm).days
            except Exception:
                los = 0
        los = max(0, los)
        los_list.append(los)

        n_dx = sum(1 for i in range(1, 11) if str(row.get("ICD9_DGNS_CD_%d" % i, "") or "").strip())
        ndx_list.append(n_dx)

        n_proc = sum(1 for i in range(1, 7) if str(row.get("ICD9_PRCDR_CD_%d" % i, "") or "").strip()) + \
                 sum(1 for i in range(1, 46) if str(row.get("HCPCS_CD_%d" % i, "") or "").strip())
        nproc_list.append(n_proc)

        primary = str(row.get("ICD9_DGNS_CD_1", "") or "").strip()
        chap_counter[icd9_chapter(primary)] += 1

    amount_sorted = sorted(amounts)
    if len(amount_sorted) > downsample and downsample >= 2:
        amount_sorted = [amount_sorted[round(i * (len(amount_sorted) - 1) / (downsample - 1))] for i in range(downsample)]

    amount_block = {
        "kind": "continuous",
        "sorted": amount_sorted,
        "n": len(amounts),
        "mean": statistics.mean(amounts) if amounts else 0.0,
        "median": statistics.median(amounts) if amounts else 0.0,
    }

    los_block = {
        "kind": "discrete",
        "dist": _norm_counter(Counter(los_list)),
        "mean": statistics.mean(los_list) if los_list else 0.0,
        "n": len(los_list),
    }

    ndx_block = {
        "kind": "discrete",
        "dist": _norm_counter(Counter(ndx_list)),
    }

    nproc_block = {
        "kind": "discrete",
        "dist": _norm_counter(Counter(nproc_list)),
    }

    chapter_block = {
        "kind": "categorical",
        "dist": _norm_counter(chap_counter),
    }

    ages = []
    female = 0
    total_b = 0
    for b in beneficiary_rows:
        total_b += 1
        bd = str(b.get("BENE_BIRTH_DT", "") or "").strip()
        if len(bd) >= 4 and bd[:4].isdigit():
            ages.append(2008 - int(bd[:4]))
        if str(b.get("BENE_SEX_IDENT_CD", "") or "").strip() == "2":
            female += 1

    age_block = {
        "kind": "discrete",
        "dist": _norm_counter(Counter(ages)),
        "mean": statistics.mean(ages) if ages else 0.0,
    }

    sex_block = {
        "female_frac": female / total_b if total_b else 0.0,
    }

    return {
        "provenance": provenance,
        "dx_chapters": list(DX_CHAPTERS),
        "amount": amount_block,
        "los": los_block,
        "n_dx": ndx_block,
        "n_proc": nproc_block,
        "dx_chapter": chapter_block,
        "age": age_block,
        "sex": sex_block,
    }


def save_reference(ref, path):
    with open(path, "w") as f:
        json.dump(ref, f, indent=1)


def load_reference(path):
    with open(path) as f:
        return json.load(f)
