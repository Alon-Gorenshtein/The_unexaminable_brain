"""Mortality model with cross-validation folds assigned by patient, next to the folds assigned by ICU stay.

The audit-subset model (lib_audit.fit_oof) assigns folds by ICU stay, so a patient with more than one stay can
contribute a stay to the training folds and another to the test fold. Here the same model, the same design matrix
and the same selected examination (lowest eye-plus-motor rule, as in 62_audit_table.py) are fitted again with whole
patients kept in one fold. Nothing already reported is changed: the stay-fold AUROC of every strategy is asserted
equal to the one in audit.json before the patient-fold AUROC is written beside it."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import INT, REV
from lib_audit import add_stay_covariates, add_totals, boot_auc_diff, cv_splits, fit_oof, selected_frames
from lib_cv_checks import patients_in_multiple_folds

TOL = 1e-9
audit = json.loads((REV / "audit.json").read_text())

# ---- who the patients are: the acute-brain-injury first-ICU-stay cohort, 1 row per stay ----
coh = pd.read_parquet(INT / "cohort.parquet", columns=["subject_id", "stay_id", "first_stay", "phenotype"])
coh = coh[coh["first_stay"] & (coh["phenotype"] != "comparator")]
assert coh["stay_id"].is_unique
subject_of = coh.set_index("stay_id")["subject_id"]

# ---- the same design as 62_audit_table.py for the primary rule ----
official = pd.read_parquet(REV / "official_first_day_gcs.parquet")[["stay_id", "gcs_min"]]
STRATS = [("official", "tg_official"), ("reconstructed", "tg_derived"), ("impute1", "tg_impute1"),
          ("gcs_t", "tg_gcs_t"), ("brennan", "tg_brennan")]
w = add_totals(add_stay_covariates(selected_frames(24)["lowest_em"]))
w = w.merge(official, on="stay_id", how="left").rename(columns={"gcs_min": "tg_official"})
assert w["stay_id"].isin(subject_of.index).all(), "a selected stay is missing from the cohort"
saved = pd.read_parquet(REV / "audit_oof_lowest_em.parquet")

res, out = {}, {}
for slug, col in STRATS:
    ids, yy, p_stay = fit_oof(w, col)
    au_stay = roc_auc_score(yy, p_stay)
    assert abs(au_stay - audit[f"{slug}_lowest_em_auroc"]) < TOL, f"stay-fold AUROC moved for {slug}: {au_stay}"
    assert np.abs(saved.loc[ids, f"p_{slug}"].values - p_stay).max() < TOL, f"stay-fold predictions moved for {slug}"
    ids_p, y_p, p_pat = fit_oof(w, col, groups=subject_of)
    assert np.array_equal(ids, ids_p) and np.array_equal(yy, y_p), f"stay set changed with the folds for {slug}"
    res[slug] = (ids, yy, p_stay, p_pat)
    au_pat = roc_auc_score(yy, p_pat)
    out.update({f"{slug}_auroc_stay_folds": float(au_stay), f"{slug}_auroc_patient_folds": float(au_pat),
                f"{slug}_diff": float(au_pat - au_stay)})

ids0, y0 = res["official"][0], res["official"][1]
for slug, (ids, yy, _, _) in res.items():
    assert np.array_equal(ids, ids0) and np.array_equal(yy, y0), f"stay sets differ for {slug}"

# ---- the folds themselves: no patient in two folds, fold sizes near 20% ----
subj = subject_of.loc[ids0]


def fold_frame(groups):
    fold = np.zeros(len(y0), dtype=int)
    for k, (_, te) in enumerate(cv_splits(y0, groups)):
        fold[te] = k
    return pd.DataFrame({"subject_id": subj.values, "fold": fold})


by_stay, by_patient = fold_frame(None), fold_frame(subj.values)
share = by_patient["fold"].value_counts(normalize=True)
assert patients_in_multiple_folds(by_patient) == 0, "patient-grouped folds split a patient"
assert 0.15 < share.min() and share.max() < 0.25, "patient-grouped fold sizes are far from 20%"
out.update(n_stays=int(len(ids0)), n_patients=int(subj.nunique()),
           patients_in_multiple_folds_stay_folds=patients_in_multiple_folds(by_stay),
           patients_in_multiple_folds_patient_folds=patients_in_multiple_folds(by_patient),
           patient_fold_pct_min=float(100 * share.min()), patient_fold_pct_max=float(100 * share.max()))

# ---- paired Brennan-minus-official difference with patient-grouped folds, stay-level bootstrap as in audit.json ----
m_, lo, hi = boot_auc_diff(y0, res["brennan"][3], res["official"][3])
out.update(brennan_minus_official_patient_folds=float(out["brennan_auroc_patient_folds"] - out["official_auroc_patient_folds"]),
           brennan_minus_official_patient_folds_mean=m_, brennan_minus_official_patient_folds_lo=float(lo),
           brennan_minus_official_patient_folds_hi=float(hi),
           brennan_minus_official_stay_folds=float(out["brennan_auroc_stay_folds"] - out["official_auroc_stay_folds"]))
out["max_abs_diff"] = float(max(abs(out[f"{s}_diff"]) for s, _ in STRATS))

(REV / "mortality_cv_by_patient.json").write_text(json.dumps(out, indent=1, allow_nan=False))
print(json.dumps(out, indent=1, allow_nan=False))
