"""Aim 3 - reproducibility audit: how 4 GCS-verbal handling strategies change a
mortality model. Built on the worst (lowest eye+motor) GCS triple in the first 24h."""
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from config import INT, TAB, ICU, IT, EYE_SCORE, MOTOR_SCORE
from lib_gcs_strategies import total_gcs

STRATS = ["drop", "impute1", "gcs_t", "kramer"]
RNG = np.random.RandomState(0)

cohort = pd.read_parquet(INT / "cohort.parquet")
cohort = cohort[cohort.first_stay]
neuro = cohort[cohort.phenotype != "comparator"].copy()
icu = pd.read_csv(ICU / "icustays.csv.gz", usecols=["stay_id", "intime"], parse_dates=["intime"])
chart = pd.read_parquet(INT / "chart_neuro.parquet").merge(icu, on="stay_id")
chart["hr"] = (chart["charttime"] - chart["intime"]).dt.total_seconds() / 3600
d1 = chart[(chart.hr >= 0) & (chart.hr <= 24)]

eye = d1[d1.itemid == IT["gcs_eye"]].assign(eye=lambda x: x["value"].map(EYE_SCORE))
mot = d1[d1.itemid == IT["gcs_motor"]].assign(motor=lambda x: x["value"].map(MOTOR_SCORE))
vrb = d1[d1.itemid == IT["gcs_verbal"]].rename(columns={"value": "verbal_raw"})
trip = (eye[["stay_id", "charttime", "eye"]]
        .merge(mot[["stay_id", "charttime", "motor"]], on=["stay_id", "charttime"])
        .merge(vrb[["stay_id", "charttime", "verbal_raw"]], on=["stay_id", "charttime"]))
trip = trip.dropna(subset=["eye", "motor", "verbal_raw"])
trip["em"] = trip["eye"] + trip["motor"]
trip = trip.sort_values(["stay_id", "em", "charttime"])
worst = trip.groupby("stay_id").first().reset_index()

X = neuro.merge(worst, on="stay_id", how="inner").set_index("stay_id")
inf = pd.read_parquet(INT / "infusions.parquet")
vent = pd.read_parquet(INT / "vent.parquet")
X["vaso"] = X.index.isin(inf[inf.cls == "vasopressor"].stay_id).astype(int)
X["vent"] = X.index.isin(vent.stay_id).astype(int)
X["y"] = X["hospital_expire_flag"].astype(int)
X["nonassess"] = X["verbal_raw"].eq("No Response-ETT")
print("Aim-3 analytic cohort:", len(X),
      "| %% non-assessable worst-exam: %.1f" % (100 * X["nonassess"].mean()))


def boot_auc_ci(y, p, B=500):
    aucs = []
    n = len(y)
    idx = np.arange(n)
    for _ in range(B):
        s = RNG.choice(idx, n, replace=True)
        if y[s].sum() == 0 or y[s].sum() == len(s):
            continue
        aucs.append(roc_auc_score(y[s], p[s]))
    return np.percentile(aucs, [2.5, 97.5])


def fit_eval(df, strat, seed=0):
    d = df.copy()
    d["tg"] = [total_gcs(dict(eye=e, motor=m, verbal_raw=v), strat)
               for e, m, v in zip(d.eye, d.motor, d.verbal_raw)]
    d = d.dropna(subset=["tg"])
    if d["y"].nunique() < 2 or len(d) < 50:
        return None, None
    feats = d[["tg", "age", "vaso", "vent"]].astype(float).values
    y = d["y"].values
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(feats, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(feats[tr], y[tr])
        oof[te] = clf.predict_proba(feats[te])[:, 1]
    lo, hi = boot_auc_ci(y, oof)
    metrics = dict(n=len(y), n_deaths=int(y.sum()),
                   auroc=round(roc_auc_score(y, oof), 4),
                   auroc_lo=round(lo, 4), auroc_hi=round(hi, 4),
                   brier=round(brier_score_loss(y, oof), 4),
                   mean_tg=round(float(d["tg"].mean()), 2))
    return metrics, pd.Series(oof, index=d.index)


rows, preds = [], {}
for s in STRATS:
    m, p = fit_eval(X, s)
    if m:
        rows.append({"phenotype": "ALL", "strategy": s, **m, "n_excluded": len(X) - m["n"]})
        preds[s] = p
for ph, dfp in X.groupby("phenotype"):
    for s in STRATS:
        m, _ = fit_eval(dfp, s)
        if m:
            rows.append({"phenotype": ph, "strategy": s, **m, "n_excluded": len(dfp) - m["n"]})
res = pd.DataFrame(rows)
res.to_csv(TAB / "table3_aim3_by_phenotype.csv", index=False)
print("\n=== Aim 3: model behavior by handling strategy (ALL neuro) ===")
print(res[res.phenotype == "ALL"].to_string(index=False))

# ---- selection effect of complete-case (drop) ----
excl = X[X["nonassess"]]
incl = X[~X["nonassess"]]
sel = dict(n_total=len(X), n_excluded=int(len(excl)),
           pct_excluded=round(100 * len(excl) / len(X), 1),
           deaths_total=int(X.y.sum()), deaths_excluded=int(excl.y.sum()),
           pct_of_deaths_excluded=round(100 * excl.y.sum() / X.y.sum(), 1),
           mortality_excluded=round(100 * excl.y.mean(), 1),
           mortality_retained=round(100 * incl.y.mean(), 1))
pd.Series(sel).to_csv(TAB / "table_selection_effect.csv")
print("\n=== Selection effect of complete-case (drop) ===")
for k, v in sel.items():
    print(f"  {k}: {v}")

# ---- reclassification impute1 vs kramer (risk tertiles) ----
if "impute1" in preds and "kramer" in preds:
    sh = preds["impute1"].index.intersection(preds["kramer"].index)
    a = pd.qcut(preds["impute1"].loc[sh], 3, labels=["low", "mid", "high"])
    b = pd.qcut(preds["kramer"].loc[sh], 3, labels=["low", "mid", "high"])
    reclass = pd.crosstab(a, b)
    moved = (a.values != b.values).mean()
    reclass.to_csv(TAB / "table_reclass_impute1_vs_kramer.csv")
    print("\nRisk-tertile reclassification impute1 -> kramer: %.1f%% move tertile" % (100 * moved))
