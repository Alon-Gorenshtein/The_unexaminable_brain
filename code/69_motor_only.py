"""Motor-only severity descriptor in the Aim 3 mortality model, on the audit subset.

Same examination selection (lowest eye-plus-motor examination in the first 24 hours), covariates,
cross-validation and bootstrap (2,000 stay-level resamples, fixed seed) as code/62_audit_table.py, so the
Brennan reference row equals the Brennan row of audit_by_rule.csv. The motor score never needs the verbal
component. Rows: all audit stays, and the stays whose selected examination was non-assessable (a separate
model fit within that subset)."""
import json

import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from config import REV
from lib_audit import add_stay_covariates, add_totals, boot_auc_ci, boot_auc_diff, fit_oof, load_first_day, select_exam

DESCRIPTORS = [("motor", "Motor score only", "motor_only"), ("brennan", "Brennan estimate (reference)", "tg_brennan")]


def selected():
    w = add_totals(add_stay_covariates(select_exam(load_first_day(24), "lowest_em")))
    w["motor_only"] = w["motor"].astype(float)
    return w


def evaluate(w):
    rows, out, preds = [], {}, {}
    for scope, sub in (("all", w), ("nonassessable", w[w["nonassess"].astype(bool)])):
        for slug, label, col in DESCRIPTORS:
            ids, y, p = fit_oof(sub, col)
            lo, hi = boot_auc_ci(y, p)
            r = dict(scope=scope, slug=slug, descriptor=label, n=len(y), deaths=int(y.sum()),
                     auroc=roc_auc_score(y, p), lo=lo, hi=hi, brier=brier_score_loss(y, p),
                     mean_value=float(sub.set_index("stay_id").loc[ids, col].mean()))
            rows.append(r)
            for k in ("n", "deaths", "auroc", "lo", "hi", "brier", "mean_value"):
                out[f"{scope}_{slug}_{k}"] = float(r[k])
            preds[(scope, slug)] = (ids, y, p)
        (ia, ya, pa), (ib, _, pb) = preds[(scope, "brennan")], preds[(scope, "motor")]
        assert (ia == ib).all()
        m, lo, hi = boot_auc_diff(ya, pa, pb)
        out.update({f"{scope}_paired_brennan_minus_motor_mean": m, f"{scope}_paired_brennan_minus_motor_lo": lo,
                    f"{scope}_paired_brennan_minus_motor_hi": hi})
    na = w[w["nonassess"].astype(bool)]
    out["nonassessable_nmb_active_n"] = int(na["nmb"].astype(int).sum())
    out["nonassessable_nmb_active_pct"] = float(100 * na["nmb"].astype(float).mean())
    return pd.DataFrame(rows), out


if __name__ == "__main__":
    table, summary = evaluate(selected())
    table.to_csv(REV / "motor_only.csv", index=False)
    (REV / "motor_only.json").write_text(json.dumps(summary, indent=1))
    print(table.round(4).to_string(index=False))
    print({k: round(v, 4) for k, v in summary.items() if "paired" in k or "nmb" in k})
