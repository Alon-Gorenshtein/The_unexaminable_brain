"""Aim 4 - sedation-aware learned imputation of the non-assessable verbal GCS.
A: can a model recover the verbal score better than the Brennan eye+motor heuristic?
B: deploy the learned imputation as a 5th handling strategy in the mortality audit.
C: where does the sedation-aware model diverge from Brennan?"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
np.seterr(all="ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, brier_score_loss
from config import TAB, INT
from lib_features import build_exam_table, FEATURES
from lib_gcs_strategies import _kramer_add

RNG = np.random.RandomState(0)
print("Building exam feature table (first 72h)...", flush=True)
ex = build_exam_table(72)
ex.to_parquet(INT / "aim4_exams.parquet", index=False)
print("Exams:", len(ex), "| assessable:", (~ex.nonassess).sum(),
      "| RASS present: %.1f%%" % (100 * ex.rass.notna().mean()), flush=True)

# ================= Part A: recovery of the verbal score =================
A = ex[~ex.nonassess].dropna(subset=["verbal_score", "eye", "motor"]).copy()
A["verbal_score"] = A["verbal_score"].astype(int)
X = A[FEATURES].values
y = A["verbal_score"].values
groups = A["stay_id"].values   # patient-level grouping: a patient's exams stay together
oof = np.zeros(len(y), dtype=int)
# StratifiedGroupKFold keeps all of a patient's examinations in the same fold (no leakage)
# while balancing the verbal-score distribution across folds.
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups):
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         max_depth=6, random_state=0)
    clf.fit(X[tr], y[tr])
    oof[te] = clf.predict(X[te])
print("Patient-level CV: %d patients across %d exams" % (len(np.unique(groups)), len(y)), flush=True)
# Brennan-implied verbal (eye+motor heuristic) and fixed=1 convention
A["em"] = A["eye"] + A["motor"]
brennan_v = A["em"].map(_kramer_add).values
fixed_v = np.ones(len(y), dtype=int)


def boot_metric(fn, y, p, B=500):
    vals = []
    idx = np.arange(len(y))
    for _ in range(B):
        s = RNG.choice(idx, len(y), replace=True)
        try:
            vals.append(fn(y[s], p[s]))
        except Exception:
            pass
    return np.percentile(vals, [2.5, 97.5])


def qwk(a, b):
    return cohen_kappa_score(a, b, weights="quadratic", labels=[1, 2, 3, 4, 5])


rows = []
for name, pred in [("Learned (sedation-aware)", oof), ("Brennan (eye+motor)", brennan_v),
                   ("Fixed verbal=1", fixed_v)]:
    acc = accuracy_score(y, pred)
    k = qwk(y, pred)
    acc_ci = boot_metric(accuracy_score, y, pred)
    qwk_ci = boot_metric(qwk, y, pred)
    rows.append(dict(method=name, n=len(y), accuracy=round(acc, 4),
                     accuracy_lo=round(acc_ci[0], 4), accuracy_hi=round(acc_ci[1], 4),
                     qwk=round(k, 4), qwk_lo=round(qwk_ci[0], 4), qwk_hi=round(qwk_ci[1], 4)))
recovery = pd.DataFrame(rows)
recovery.to_csv(TAB / "table_aim4_recovery.csv", index=False)
print("\n=== Aim 4A: verbal-score recovery on held-out assessable exams ===")
print(recovery.to_string(index=False))

# train final verbal model on ALL assessable exams for downstream imputation
final_clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                           max_depth=6, random_state=0).fit(X, y)

# ================= Part B: 5th handling strategy in the mortality audit =================
worst = (ex[ex.hr <= 24].assign(em=lambda d: d.eye + d.motor)
         .sort_values(["stay_id", "em", "charttime"])
         .groupby("stay_id").first().reset_index())
worst = worst.dropna(subset=["eye", "motor", "verbal_raw"])
# learned verbal for non-assessable worst exams
na = worst[worst.nonassess]
worst["verbal_learned"] = worst["verbal_score"]
if len(na):
    pred_v = final_clf.predict(na[FEATURES].values)
    worst.loc[worst.nonassess, "verbal_learned"] = pred_v
worst["tg_learned"] = worst["eye"] + worst["motor"] + worst["verbal_learned"]
worst["tg_brennan"] = worst.apply(
    lambda r: r.eye + r.motor + (r.verbal_score if not r.nonassess else _kramer_add(r.eye + r.motor)),
    axis=1)
# stay-level vasopressor/ventilation covariates to match the Aim 3 mortality model
_inf = pd.read_parquet(INT / "infusions.parquet")
_vent = pd.read_parquet(INT / "vent.parquet")
worst["vaso"] = worst["stay_id"].isin(_inf[_inf.cls == "vasopressor"].stay_id).astype(int)
worst["vent"] = worst["stay_id"].isin(_vent.stay_id).astype(int)


def fit_eval(df, tgcol):
    d = df.dropna(subset=[tgcol]).copy()
    feats = d[[tgcol, "age", "vaso", "vent"]].astype(float).values
    yy = d["y"].astype(int).values
    oofp = np.zeros(len(yy))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(feats, yy):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(feats[tr], yy[tr])
        oofp[te] = clf.predict_proba(feats[te])[:, 1]
    lo, hi = boot_metric(roc_auc_score, yy, oofp)
    return dict(n=len(yy), auroc=round(roc_auc_score(yy, oofp), 4),
                auroc_lo=round(lo, 4), auroc_hi=round(hi, 4),
                brier=round(brier_score_loss(yy, oofp), 4),
                mean_tg=round(float(d[tgcol].mean()), 2)), pd.Series(oofp, index=d.index)

ml, pl = fit_eval(worst, "tg_learned")
mb, pb = fit_eval(worst, "tg_brennan")
audit = pd.DataFrame([{"strategy": "Learned (sedation-aware)", **ml},
                      {"strategy": "Brennan", **mb}])
audit.to_csv(TAB / "table_aim4_audit.csv", index=False)
print("\n=== Aim 4B: learned imputation as a handling strategy (worst-exam mortality model) ===")
print(audit.to_string(index=False))
# reclassification learned vs Brennan
sh = pl.index.intersection(pb.index)
a = pd.qcut(pl.loc[sh], 3, labels=["low", "mid", "high"])
b = pd.qcut(pb.loc[sh], 3, labels=["low", "mid", "high"])
moved = (a.values != b.values).mean()
print("Reclassification learned vs Brennan: %.1f%% move tertile" % (100 * moved))

# ================= Part C: divergence from Brennan on non-assessable exams =================
nad = worst[worst.nonassess].copy()
nad["diff"] = nad["tg_learned"] - nad["tg_brennan"]
nad["deep"] = np.where(nad["rass"] <= -3, "Deep sedation (RASS<=-3)",
                       np.where(nad["rass"].notna(), "Light/none (RASS>-3)", "RASS missing"))
agr = nad.groupby("deep")["diff"].agg(["size", "mean", "std"]).round(2)
agr.to_csv(TAB / "table_aim4_agreement.csv")
overall = dict(n_nonassess=len(nad), mean_diff=round(nad["diff"].mean(), 2),
               pct_differ=round(100 * (nad["diff"] != 0).mean(), 1),
               pct_learned_lower=round(100 * (nad["diff"] < 0).mean(), 1))
print("\n=== Aim 4C: learned minus Brennan total GCS on non-assessable worst exams ===")
print(overall)
print(agr.to_string())
pd.Series(overall).to_csv(TAB / "table_aim4_agreement_overall.csv")
