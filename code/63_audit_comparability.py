"""Put every imputation strategy on the same patients, keep complete-case separate, and report
calibration where the strategies actually differ (the non-assessable subgroup)."""
import json

import pandas as pd
from sklearn.metrics import roc_auc_score

from config import REV
from lib_audit import boot_auc_ci, boot_auc_diff, calib_slope

oof = pd.read_parquet(REV / "audit_oof_lowest_em.parquet")
FULL = ["official", "reconstructed", "impute1", "gcs_t", "brennan"]
P = {s: f"p_{s}" for s in FULL + ["complete_case"]}
common = oof[[P[s] for s in FULL]].notna().all(axis=1)
na = oof["nonassess"].astype(bool)


def block(mask, slug):
    d = oof[mask & oof[P[slug]].notna()]
    y, p = d["y"].values.astype(int), d[P[slug]].values
    lo, hi = boot_auc_ci(y, p)
    return dict(n=len(d), deaths=int(y.sum()), mortality_pct=100 * y.mean(), auroc=roc_auc_score(y, p),
                lo=lo, hi=hi, mean_predicted_pct=100 * p.mean(), oe_ratio=y.mean() / p.mean(),
                calib_slope=calib_slope(y, p))


rows, out = [], {}
for group, mask in [("all", common), ("nonassessable", common & na), ("assessable", common & ~na)]:
    for s in FULL:
        rows.append(dict(group=group, slug=s, **block(mask, s)))
cc = oof[P["complete_case"]].notna()
for s in ["complete_case", "brennan", "official"]:
    rows.append(dict(group="cc", slug=s, **block(cc, s)))
m, lo, hi = boot_auc_diff(oof.loc[cc, "y"].values.astype(int), oof.loc[cc, P["complete_case"]].values,
                          oof.loc[cc, P["brennan"]].values)
res = pd.DataFrame(rows)
res.to_csv(REV / "comparability.csv", index=False)
for r in res.itertuples():
    for k in ("n", "deaths", "mortality_pct", "auroc", "lo", "hi", "mean_predicted_pct", "oe_ratio", "calib_slope"):
        v = getattr(r, k)
        out[f"{r.group}_{r.slug}_{k}"] = int(v) if k in ("n", "deaths") else float(v)
out.update(cc_diff_cc_minus_brennan_mean=m, cc_diff_cc_minus_brennan_lo=lo, cc_diff_cc_minus_brennan_hi=hi)
(REV / "comparability.json").write_text(json.dumps(out, indent=1, allow_nan=False))
print(res.round(3).to_string(index=False))
