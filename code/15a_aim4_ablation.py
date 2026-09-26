"""Aim 4 robustness analyses.

(1) Mechanical-ventilation ablation. `vent` is near-deterministic for
    the non-assessable target, and the verbal model is trained on assessable exams (where
    vent is rare) and applied to intubated exams (where vent is near-universal), risking
    circularity. We refit the verbal-recovery model WITHOUT `vent` and compare recovery and the
    downstream mortality model with the with-vent variant. -> eTable (ablation).
(2) GCS-motor-only fallback. Some intubated patients still have an interpretable motor response;
    we benchmark a motor-component-only severity descriptor against the Brennan total in the
    Aim-3 mortality model, overall and restricted to the non-assessable subset. -> Results sentence.

Reproducible from the cached aim4_exams.parquet + cohort/chart/infusion/vent parquets.
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (cohen_kappa_score, accuracy_score, roc_auc_score,
                             brier_score_loss)
from config import INT, TAB, ICU, IT, EYE_SCORE, MOTOR_SCORE
from lib_features import FEATURES, FEATURES_WITH_VENT
from lib_gcs_strategies import total_gcs, _kramer_add

RNG = np.random.RandomState(0)
ex = pd.read_parquet(INT / "aim4_exams.parquet")


def boot(fn, y, p, B=500):
    vals, idx = [], np.arange(len(y))
    for _ in range(B):
        s = RNG.choice(idx, len(y), replace=True)
        try:
            vals.append(fn(y[s], p[s]))
        except Exception:
            pass
    return np.percentile(vals, [2.5, 97.5])


def qwk(a, b):
    return cohen_kappa_score(a, b, weights="quadratic", labels=[1, 2, 3, 4, 5])


# ============== (1a) verbal-recovery ablation: with vs without `vent` ==============
A = ex[~ex.nonassess].dropna(subset=["verbal_score", "eye", "motor"]).copy()
A["verbal_score"] = A["verbal_score"].astype(int)
y = A["verbal_score"].values
groups = A["stay_id"].values
print("Assessable training set:", len(A), "exams |",
      "vent-positive: %.1f%%" % (100 * A.vent.mean()), flush=True)


def cv_recovery(feats):
    X = A[feats].values
    oof = np.zeros(len(y), dtype=int)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                             max_depth=6, random_state=0).fit(X[tr], y[tr])
        oof[te] = clf.predict(X[te])
    return oof


rec_rows = []
for label, feats in [("Without ventilation flag (reported)", FEATURES),
                     ("With ventilation flag", FEATURES_WITH_VENT)]:
    oof = cv_recovery(feats)
    acc, k = accuracy_score(y, oof), qwk(y, oof)
    acc_ci, k_ci = boot(accuracy_score, y, oof), boot(qwk, y, oof)
    rec_rows.append(dict(model=label, n=len(y), n_features=len(feats),
                         accuracy=round(acc, 4), accuracy_lo=round(acc_ci[0], 4),
                         accuracy_hi=round(acc_ci[1], 4), qwk=round(k, 4),
                         qwk_lo=round(k_ci[0], 4), qwk_hi=round(k_ci[1], 4)))

# ============== (1b) downstream mortality ablation ==============
worst = (ex[ex.hr <= 24].assign(em=lambda d: d.eye + d.motor)
         .sort_values(["stay_id", "em", "charttime"]).groupby("stay_id").first().reset_index())
worst = worst.dropna(subset=["eye", "motor", "verbal_raw"])
_inf = pd.read_parquet(INT / "infusions.parquet")
_vent = pd.read_parquet(INT / "vent.parquet")
worst["vaso"] = worst.stay_id.isin(_inf[_inf.cls == "vasopressor"].stay_id).astype(int)
worst["vent"] = worst.stay_id.isin(_vent.stay_id).astype(int)
worst["tg_brennan"] = worst.apply(
    lambda r: r.eye + r.motor + (r.verbal_score if not r.nonassess else _kramer_add(r.eye + r.motor)),
    axis=1)
na = worst[worst.nonassess]


def fit_eval(df, col):
    d = df.dropna(subset=[col]).copy()
    feats = d[[col, "age", "vaso", "vent"]].astype(float).values
    yy = d["y"].astype(int).values
    oof = np.zeros(len(yy))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(feats, yy):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(feats[tr], yy[tr])
        oof[te] = clf.predict_proba(feats[te])[:, 1]
    lo, hi = boot(roc_auc_score, yy, oof)
    return dict(n=len(yy), deaths=int(yy.sum()), auroc=round(roc_auc_score(yy, oof), 4),
                auroc_lo=round(lo, 4), auroc_hi=round(hi, 4),
                brier=round(brier_score_loss(yy, oof), 4),
                mean_tg=round(float(d[col].mean()), 2))


down_rows = []
for label, feats in [("Without ventilation flag (reported)", FEATURES),
                     ("With ventilation flag", FEATURES_WITH_VENT)]:
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         max_depth=6, random_state=0).fit(A[feats].values, y)
    worst["tg_learned"] = worst.eye + worst.motor + worst.verbal_score
    if len(na):
        worst.loc[worst.nonassess, "tg_learned"] = (
            worst.loc[worst.nonassess, "eye"] + worst.loc[worst.nonassess, "motor"]
            + clf.predict(na[feats].values))
    m = fit_eval(worst, "tg_learned")
    down_rows.append(dict(model=label, **m))

abl = pd.DataFrame(rec_rows)
abl_down = pd.DataFrame(down_rows)
abl.to_csv(TAB / "table_aim4_vent_ablation_recovery.csv", index=False)
abl_down.to_csv(TAB / "table_aim4_vent_ablation_downstream.csv", index=False)
print("\n=== Aim 4 ventilation-flag ablation: verbal recovery ===")
print(abl.to_string(index=False))
print("\n=== Aim 4 ventilation-flag ablation: downstream mortality model ===")
print(abl_down.to_string(index=False))

# ============== (2) GCS-motor-only fallback benchmark (Aim-3 mortality model) ==============
cohort = pd.read_parquet(INT / "cohort.parquet"); cohort = cohort[cohort.first_stay]
neuro = cohort[cohort.phenotype != "comparator"].copy()
icu = pd.read_csv(ICU / "icustays.csv.gz", usecols=["stay_id", "intime"], parse_dates=["intime"])
chart = pd.read_parquet(INT / "chart_neuro.parquet").merge(icu, on="stay_id")
chart["hr"] = (chart.charttime - chart.intime).dt.total_seconds() / 3600
d1 = chart[(chart.hr >= 0) & (chart.hr <= 24)]
eye = d1[d1.itemid == IT["gcs_eye"]].assign(eye=lambda x: x["value"].map(EYE_SCORE))
mot = d1[d1.itemid == IT["gcs_motor"]].assign(motor=lambda x: x["value"].map(MOTOR_SCORE))
vrb = d1[d1.itemid == IT["gcs_verbal"]].rename(columns={"value": "verbal_raw"})
trip = (eye[["stay_id", "charttime", "eye"]].merge(mot[["stay_id", "charttime", "motor"]], on=["stay_id", "charttime"])
        .merge(vrb[["stay_id", "charttime", "verbal_raw"]], on=["stay_id", "charttime"])).dropna(subset=["eye", "motor", "verbal_raw"])
trip["em"] = trip.eye + trip.motor
w2 = trip.sort_values(["stay_id", "em", "charttime"]).groupby("stay_id").first().reset_index()
X = neuro.merge(w2, on="stay_id", how="inner").set_index("stay_id")
X["vaso"] = X.index.isin(_inf[_inf.cls == "vasopressor"].stay_id).astype(int)
X["vent"] = X.index.isin(_vent.stay_id).astype(int)
X["y"] = X["hospital_expire_flag"].astype(int)
X["nonassess"] = X.verbal_raw.eq("No Response-ETT")
X["motor_only"] = X.motor.astype(float)
X["tg_kramer"] = [total_gcs(dict(eye=e, motor=m, verbal_raw=v), "kramer")
                  for e, m, v in zip(X.eye, X.motor, X.verbal_raw)]
# active NMB among non-assessable (motor interpretability check)
nmb_na = ex[ex.nonassess & (ex.hr <= 24)].nmb.mean()

mo_rows = [dict(scope="All neuro", strategy="GCS-motor-only", **fit_eval(X.assign(y=X.y), "motor_only")),
           dict(scope="All neuro", strategy="Brennan total (reference)", **fit_eval(X, "tg_kramer")),
           dict(scope="Non-assessable only", strategy="GCS-motor-only", **fit_eval(X[X.nonassess], "motor_only"))]
mo = pd.DataFrame(mo_rows)
mo.to_csv(TAB / "table_aim3_motor_only.csv", index=False)
print("\n=== GCS-motor-only fallback benchmark (Aim-3 mortality model) ===")
print("Active NMB among non-assessable worst exams: %.2f%% (motor essentially always interpretable)"
      % (100 * nmb_na))
print(mo.to_string(index=False))
