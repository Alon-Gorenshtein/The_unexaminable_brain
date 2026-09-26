"""Subject-level cluster bootstrap, as a check on the stay-level intervals.

The cohort is the first ICU stay of each hospitalization, so a patient (subject_id) can contribute more than one
stay. The intervals in aim4.json cluster all examinations of a stay together, and the AUROC intervals in audit.json
resample one selected examination per stay; neither treats the stays of one patient as dependent. Here whole
patients are resampled with replacement and every stay (and every examination) of a resampled patient is kept.
Nothing already reported is changed: every point estimate is asserted equal to the one in aim4.json / audit.json,
and the stay-level interval endpoints are copied beside the subject-level ones."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import INT, REV
from lib_audit import add_stay_covariates, add_totals, fit_oof, selected_frames
from lib_cluster import (N_RESAMPLES, SEED, acc_from_confusion, cluster_auc_ci, cluster_auc_diff_ci, cluster_ci, cluster_diff_ci,
                         patient_confusions, qwk_from_confusion)

TOL = 1e-9
aim4 = json.loads((REV / "aim4.json").read_text())
audit = json.loads((REV / "audit.json").read_text())
out = {"seed": SEED, "n_resamples": N_RESAMPLES}

# ---- who the patients are: the acute-brain-injury first-ICU-stay cohort, 1 row per stay ----
coh = pd.read_parquet(INT / "cohort.parquet", columns=["subject_id", "stay_id", "first_stay", "phenotype"])
coh = coh[coh["first_stay"] & (coh["phenotype"] != "comparator")]
assert coh["stay_id"].is_unique
stays_per_subject = coh.groupby("subject_id").size()
out.update(n_stays_total=int(len(coh)), n_subjects_total=int(len(stays_per_subject)),
           n_subjects_with_multiple_stays=int((stays_per_subject > 1).sum()),
           n_stays_of_multi_stay_subjects=int(stays_per_subject[stays_per_subject > 1].sum()),
           max_stays_per_subject=int(stays_per_subject.max()))
subject_of = coh.set_index("stay_id")["subject_id"]

# ---- Aim 4: quadratic weighted kappa of the verbal-score estimate, examinations clustered by patient ----
oof = pd.read_parquet(REV / "aim4_oof.parquet")
oof["subject_id"] = oof["stay_id"].map(subject_of)
assert oof["subject_id"].notna().all(), "an Aim 4 stay is missing from the cohort"
y, g = oof["y"].values, oof["subject_id"].values
out.update(aim4_n_exams=int(len(oof)), aim4_n_stays=int(oof["stay_id"].nunique()),
           aim4_n_subjects=int(oof["subject_id"].nunique()))
assert out["aim4_n_exams"] == aim4["n_exams"] and out["aim4_n_stays"] == aim4["n_patients"]
M = {}
for m in ("learned", "learned_no_rass", "brennan", "fixed1"):
    M[m] = patient_confusions(y, oof[m].values, g)
    C = M[m].sum(0)
    q, a = qwk_from_confusion(C), acc_from_confusion(C)
    assert abs(q - aim4[f"{m}_qwk"]) < TOL and abs(a - aim4[f"{m}_acc"]) < TOL, f"Aim 4 point estimate moved for {m}"
    ql, qh = cluster_ci(M[m], qwk_from_confusion, B=N_RESAMPLES, seed=SEED)
    al, ah = cluster_ci(M[m], acc_from_confusion, B=N_RESAMPLES, seed=SEED)
    out.update({f"aim4_subject_{m}_qwk": float(q), f"aim4_subject_{m}_qwk_lo": float(ql),
                f"aim4_subject_{m}_qwk_hi": float(qh), f"aim4_subject_{m}_acc": float(a),
                f"aim4_subject_{m}_acc_lo": float(al), f"aim4_subject_{m}_acc_hi": float(ah)})
    for stat in ("qwk", "acc"):
        for end in ("lo", "hi"):
            out[f"aim4_stay_{m}_{stat}_{end}"] = aim4[f"{m}_{stat}_{end}"]
lo, hi = cluster_diff_ci(M["learned"], M["brennan"], qwk_from_confusion, B=N_RESAMPLES, seed=SEED)
out.update(aim4_subject_qwk_learned_minus_brennan_lo=float(lo), aim4_subject_qwk_learned_minus_brennan_hi=float(hi),
           aim4_stay_qwk_learned_minus_brennan_lo=aim4["qwk_learned_minus_brennan_lo"],
           aim4_stay_qwk_learned_minus_brennan_hi=aim4["qwk_learned_minus_brennan_hi"])

# ---- Audit: out-of-fold mortality AUROC, primary rule (lowest eye-plus-motor examination), same calls as 62_audit_table.py ----
official = pd.read_parquet(REV / "official_first_day_gcs.parquet")[["stay_id", "gcs_min"]]
STRATS = [("official", "tg_official"), ("reconstructed", "tg_derived"), ("impute1", "tg_impute1"),
          ("gcs_t", "tg_gcs_t"), ("brennan", "tg_brennan")]
w = add_totals(add_stay_covariates(selected_frames(24)["lowest_em"]))
w = w.merge(official, on="stay_id", how="left").rename(columns={"gcs_min": "tg_official"})
saved = pd.read_parquet(REV / "audit_oof_lowest_em.parquet")
res = {}
for slug, col in STRATS:
    ids, yy, p = fit_oof(w, col)
    au = roc_auc_score(yy, p)
    assert abs(au - audit[f"{slug}_lowest_em_auroc"]) < TOL, f"audit AUROC moved for {slug}: {au}"
    ref = saved.loc[ids, f"p_{slug}"].values                # the predictions 62_audit_table.py wrote
    assert np.abs(ref - p).max() < TOL, f"out-of-fold predictions differ from audit_oof_lowest_em.parquet for {slug}"
    res[slug] = (ids, yy, p, au)
ids0, y0 = res["official"][0], res["official"][1]
for slug, (ids, yy, _, _) in res.items():
    assert np.array_equal(ids, ids0) and np.array_equal(yy, y0), f"stay sets differ for {slug}"
subj = subject_of.loc[ids0].values
out.update(audit_n_stays=int(len(ids0)), audit_n_subjects=int(len(np.unique(subj))),
           audit_n_subjects_with_multiple_stays=int((pd.Series(subj).value_counts() > 1).sum()))
assert out["audit_n_stays"] == audit["paired_brennan_minus_official_n"]
for slug, (ids, yy, p, au) in res.items():
    lo, hi = cluster_auc_ci(yy, p, subj)
    slo, shi = cluster_auc_ci(yy, p, ids)                    # clustering by stay must give back the stay-level interval
    assert abs(slo - audit[f"{slug}_lowest_em_lo"]) < TOL and abs(shi - audit[f"{slug}_lowest_em_hi"]) < TOL, slug
    out.update({f"audit_subject_{slug}_auroc": float(au), f"audit_subject_{slug}_lo": float(lo),
                f"audit_subject_{slug}_hi": float(hi),
                f"audit_stay_{slug}_lo": audit[f"{slug}_lowest_em_lo"], f"audit_stay_{slug}_hi": audit[f"{slug}_lowest_em_hi"]})
m_, lo, hi = cluster_auc_diff_ci(y0, res["brennan"][2], res["official"][2], subj)
sm, slo, shi = cluster_auc_diff_ci(y0, res["brennan"][2], res["official"][2], ids0)
assert max(abs(sm - audit["paired_brennan_minus_official_mean"]), abs(slo - audit["paired_brennan_minus_official_lo"]),
           abs(shi - audit["paired_brennan_minus_official_hi"])) < TOL, "paired stay-level interval not reproduced"
out.update(audit_paired_brennan_minus_official_point=float(res["brennan"][3] - res["official"][3]),
           audit_subject_paired_brennan_minus_official_mean=float(m_),
           audit_subject_paired_brennan_minus_official_lo=float(lo),
           audit_subject_paired_brennan_minus_official_hi=float(hi),
           audit_stay_paired_brennan_minus_official_mean=audit["paired_brennan_minus_official_mean"],
           audit_stay_paired_brennan_minus_official_lo=audit["paired_brennan_minus_official_lo"],
           audit_stay_paired_brennan_minus_official_hi=audit["paired_brennan_minus_official_hi"])

(REV / "cluster_sensitivity.json").write_text(json.dumps(out, indent=1, allow_nan=False))
print(json.dumps(out, indent=1, allow_nan=False))
