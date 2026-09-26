"""Time-aligned eICU check: total GCS charted after ventilation start. A total of 12 or more
requires a verbal score of at least 2, which an intubated patient cannot give (maximum
eye+motor is 10, plus verbal 1 is 11). Uses no imputation.

Windows are defined in lib_eicu.WINDOWS. Each stay contributes the record nearest the anchor
(earliest record after it, latest record before it); pre_6h_earliest is the record furthest
before the anchor and is kept only to show how much that choice moves the reference.
Paired variant: stays with a nearest pre-window record and a 0-6 h post-window record."""
import json

import pandas as pd

from config import REV
from lib_eicu import EICU, NEURO_DX_TERMS, WINDOWS, load_anchor, load_gcs, window_records, wilson_pct

gcs = load_gcs()
tv = load_anchor()
pat = pd.read_csv(f"{EICU}/patient.csv.gz", usecols=["patientunitstayid", "apacheadmissiondx"])
pat["neuro"] = pat.apacheadmissiondx.fillna("").str.lower().map(lambda s: any(t in s for t in NEURO_DX_TERMS))
neuro_ids = set(pat.loc[pat.neuro, "patientunitstayid"])

out = {}
recs = {name: window_records(gcs, tv, name) for name in WINDOWS}
for sub, keep in [("all", None), ("neuro", neuro_ids)]:
    for name, f in recs.items():
        if keep is not None:
            f = f[f.stay.isin(keep)]
        n = len(f)
        for label, k in [("ge12", int((f.total >= 12).sum())), ("eq15", int((f.total == 15).sum()))]:
            pct, l, h = wilson_pct(k, n)
            out.update({f"{sub}_{name}_n": n, f"{sub}_{name}_{label}_n": k, f"{sub}_{name}_{label}_pct": pct,
                        f"{sub}_{name}_{label}_lo": l, f"{sub}_{name}_{label}_hi": h})

    # paired: same stays before (nearest) and after (nearest) the anchor; counts only
    pre, post = recs["pre_6h"], recs["w0_6h"]
    if keep is not None:
        pre, post = pre[pre.stay.isin(keep)], post[post.stay.isin(keep)]
    pr = pre.set_index("stay").total.ge(12).rename("pre")
    po = post.set_index("stay").total.ge(12).rename("post")
    pair = pd.concat([pr, po], axis=1, join="inner")
    n = len(pair)
    pre_k, post_k = int(pair.pre.sum()), int(pair.post.sum())
    out.update({
        f"{sub}_paired_n": n,
        f"{sub}_paired_pre_ge12_n": pre_k, f"{sub}_paired_pre_ge12_pct": 100 * pre_k / n,
        f"{sub}_paired_post_ge12_n": post_k, f"{sub}_paired_post_ge12_pct": 100 * post_k / n,
        f"{sub}_paired_pre_only_ge12_n": int((pair.pre & ~pair.post).sum()),
        f"{sub}_paired_post_only_ge12_n": int((~pair.pre & pair.post).sum()),
    })
(REV / "eicu_time_aligned.json").write_text(json.dumps(out, indent=1, allow_nan=False))
print(json.dumps({k: round(v, 1) for k, v in out.items() if k.endswith("_pct") or k.endswith("_n")}, indent=1))
