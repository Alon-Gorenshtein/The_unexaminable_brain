"""Aim 4 (secondary, moves to the supplement). Fixes: (1) patient-level intervals, (2) RASS taken
from the selected examination's own look-back window, not borrowed from another, (3) RASS missingness
reported, (4) folds grouped by patient."""
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold

from config import INT, REV
from lib_audit import (add_stay_covariates, add_totals, boot_auc_ci, boot_auc_diff, fit_oof,
                       load_exam_table, load_first_day, select_exam)
from lib_cluster import (acc_from_confusion, cluster_ci, cluster_diff_ci, exam_level_ci, patient_confusions,
                         qwk_from_confusion)
from lib_cv_checks import patients_in_multiple_folds
from lib_features import FEATURES
from lib_gcs_strategies import _kramer_add

ex = load_exam_table()                       # corrected: keeps examinations with no eye opening
ex_legacy = load_exam_table(legacy=True)     # the submitted table, for reconciliation only
A = ex[~ex.nonassess].dropna(subset=["verbal_score", "eye", "motor"]).copy()
A["verbal_score"] = A.verbal_score.astype(int)
A["em"] = A.eye + A.motor
coh = pd.read_parquet(INT / "cohort.parquet", columns=["subject_id", "stay_id", "first_stay", "phenotype"])
subject_of = coh[coh["first_stay"] & (coh["phenotype"] != "comparator")].set_index("stay_id")["subject_id"]
A["subject_id"] = A.stay_id.map(subject_of)
assert A.subject_id.notna().all(), "a secondary-model stay is missing from the cohort"
y, g, gs = A.verbal_score.values, A.stay_id.values, A.subject_id.values   # g: bootstrap unit (stay); gs: fold unit (patient)
GBM = dict(max_iter=300, learning_rate=0.08, max_depth=6, random_state=0)
FOLD = np.zeros(len(y), dtype=int)


def cv_predict(cols):
    Xc, oof = A[cols].values, np.zeros(len(y), dtype=int)
    for k, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xc, y, gs)):
        FOLD[te] = k
        oof[te] = HistGradientBoostingClassifier(**GBM).fit(Xc[tr], y[tr]).predict(Xc[te])
    assert patients_in_multiple_folds(pd.DataFrame({"subject_id": gs, "fold": FOLD})) == 0, \
        "a patient is in both the training and the test fold"
    return oof


preds = {"learned": cv_predict(FEATURES),
         "learned_no_rass": cv_predict([c for c in FEATURES if c != "rass"]),
         "brennan": A.em.map(_kramer_add).values.astype(int),
         "fixed1": np.ones(len(y), dtype=int)}
pd.DataFrame({"stay_id": g, "subject_id": gs, "fold": FOLD, "y": y, **preds}).to_parquet(REV / "aim4_oof.parquet", index=False)

# n_patients counts ICU stays (the cluster unit here is stay_id), not patients; the key name is kept because the digest and figures read it
out, rows, M = dict(n_exams=int(len(y)), n_patients=int(len(np.unique(g))),
                    cv_group_unit="subject_id", n_cv_subjects=int(len(np.unique(gs)))), [], {}
for name, p in preds.items():
    M[name] = patient_confusions(y, p, g)
    C = M[name].sum(0)
    q, a = qwk_from_confusion(C), acc_from_confusion(C)
    ql, qh = cluster_ci(M[name], qwk_from_confusion)
    al, ah = cluster_ci(M[name], acc_from_confusion)
    qel, qeh = exam_level_ci(C, qwk_from_confusion)
    rows.append(dict(method=name, qwk=q, qwk_lo=ql, qwk_hi=qh, acc=a, acc_lo=al, acc_hi=ah,
                     qwk_exam_lo=qel, qwk_exam_hi=qeh))
    for k, v in rows[-1].items():
        if k != "method":
            out[f"{name}_{k}"] = float(v)
pd.DataFrame(rows).to_csv(REV / "aim4_recovery.csv", index=False)
lo, hi = cluster_diff_ci(M["learned"], M["brennan"], qwk_from_confusion)
out.update(qwk_learned_minus_brennan_lo=float(lo), qwk_learned_minus_brennan_hi=float(hi))

# ---- reconciliation: the eye-and-motor rule and fixed imputation on the submitted (legacy) examinations ----
Al = ex_legacy[~ex_legacy.nonassess].dropna(subset=["verbal_score", "eye", "motor"]).copy()
yl, gl = Al.verbal_score.astype(int).values, Al.stay_id.values
for name, pl in [("brennan", (Al.eye + Al.motor).map(_kramer_add).values.astype(int)),
                 ("fixed1", np.ones(len(yl), dtype=int))]:
    Cl = patient_confusions(yl, pl, gl).sum(0)
    out[f"legacy_{name}_qwk"] = float(qwk_from_confusion(Cl))
    out[f"legacy_{name}_acc"] = float(acc_from_confusion(Cl))

# ---- downstream mortality model with the corrected selected-exam features ----
final = HistGradientBoostingClassifier(**GBM).fit(A[FEATURES].values, y)
w = add_stay_covariates(select_exam(load_first_day(24), "lowest_em"))
w["verbal_learned"] = w["verbal_score"]
m = w["nonassess"].values.astype(bool)
w.loc[m, "verbal_learned"] = final.predict(w.loc[m, FEATURES].values)
w["tg_learned"] = w["em"] + w["verbal_learned"]
w = add_totals(w)
ids, yy, p_l = fit_oof(w, "tg_learned")
ids_b, _, p_b = fit_oof(w, "tg_brennan")
assert len(ids) == len(ids_b) and (ids == ids_b).all()   # the paired bootstrap below needs both predictions on the same stays in the same order
lo_l, hi_l = boot_auc_ci(yy, p_l)
lo_b, hi_b = boot_auc_ci(yy, p_b)
from sklearn.metrics import roc_auc_score
out.update(learned_downstream_auroc=float(roc_auc_score(yy, p_l)), learned_downstream_lo=float(lo_l),
           learned_downstream_hi=float(hi_l), learned_downstream_mean_gcs=float(w["tg_learned"].mean()),
           brennan_downstream_auroc=float(roc_auc_score(yy, p_b)), brennan_downstream_lo=float(lo_b),
           brennan_downstream_hi=float(hi_b))
dm, dlo, dhi = boot_auc_diff(yy, p_l, p_b)
out.update(learned_minus_brennan_auroc_mean=float(dm), learned_minus_brennan_auroc_lo=float(dlo),
           learned_minus_brennan_auroc_hi=float(dhi))


# ---- RASS: how many selected exams the submitted analysis filled from another exam ----
def n_borrowed(d):
    # .first() here reproduces the submitted code's behaviour (first non-null RASS of the stay) to COUNT
    # the borrowed values; it must not be used for analysis.
    naive = d.sort_values(["stay_id", "em", "charttime"]).groupby("stay_id").first().rass
    true = select_exam(d, "lowest_em").set_index("stay_id").rass
    return int((naive.reindex(true.index).fillna(-99) != true.fillna(-99)).sum())


d1 = load_first_day(24)
out["n_rass_borrowed_v1"] = n_borrowed(load_first_day(24, legacy=True))     # the submitted population
out["n_rass_borrowed_corrected_pop"] = n_borrowed(d1)

# ---- RASS missingness ----
ex["rass_missing"] = ex.rass.isna()
sel = select_exam(d1, "lowest_em")
out["rass_missing_all_pct"] = float(100 * ex.rass_missing.mean())
out["rass_missing_selected_pct"] = float(100 * sel.rass.isna().mean())
out["rass_missing_selected_na_pct"] = float(100 * sel[sel.nonassess].rass.isna().mean())
out["rass_missing_selected_assessable_pct"] = float(100 * sel[~sel.nonassess].rass.isna().mean())
tab = pd.concat([ex.groupby("nonassess").rass_missing.mean().mul(100).rename(lambda k: f"nonassess={k}"),
                 ex.groupby("phenotype").rass_missing.mean().mul(100),
                 ex.groupby("vent").rass_missing.mean().mul(100).rename(lambda k: f"vent={k}")]).rename("rass_missing_pct")
tab.to_csv(REV / "aim4_rass_missing.csv")
(REV / "aim4.json").write_text(json.dumps(out, indent=1, allow_nan=False))
print(json.dumps(out, indent=1, allow_nan=False))
