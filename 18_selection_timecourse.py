"""(A) Selection/missingness model: predictors of a non-assessable worst first-day verbal GCS, and
standardized mean differences (SMD) between complete-case retained vs excluded patients (the
'not missing at random' argument). (B) Clearer time-course: % of ventilated patients with NO
assessable verbal GCS at all in the first 24/48/72 h."""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
import statsmodels.api as sm
from config import INT, TAB, ICU, IT

# ---------- worst-exam cohort with predictors ----------
ex = pd.read_parquet(INT / "aim4_exams.parquet")
d1 = ex[ex.hr <= 24].dropna(subset=["eye", "motor"]).copy()
d1["em"] = d1.eye + d1.motor
worst = d1.sort_values(["stay_id", "em", "charttime"]).groupby("stay_id").first().reset_index()
inf = pd.read_parquet(INT / "infusions.parquet"); vent = pd.read_parquet(INT / "vent.parquet")
worst["vent"] = worst.stay_id.isin(vent.stay_id).astype(int)
worst["sed"] = worst.stay_id.isin(inf[inf.cls == "sedative"].stay_id).astype(int)
worst["vaso"] = worst.stay_id.isin(inf[inf.cls == "vasopressor"].stay_id).astype(int)
worst["na"] = worst.nonassess.astype(int)
worst["died"] = worst.y.astype(int)

# ---------- (A1) logistic: predictors of non-assessability ----------
dum = pd.get_dummies(worst.phenotype, prefix="ph")
ref = "ph_AIS"
Xc = [c for c in dum.columns if c != ref]
X = sm.add_constant(pd.concat([dum[Xc], worst[["age", "vent", "sed", "vaso", "died"]].reset_index(drop=True)], axis=1).astype(float))
res = sm.Logit(worst.na.values, X).fit(disp=0)
rows = []
for c in Xc + ["age", "vent", "sed", "vaso", "died"]:
    rows.append(dict(predictor=c.replace("ph_", "") + (" vs AIS" if c.startswith("ph_") else ""),
                     OR=round(np.exp(res.params[c]), 2),
                     lo=round(np.exp(res.conf_int().loc[c, 0]), 2),
                     hi=round(np.exp(res.conf_int().loc[c, 1]), 2), p=res.pvalues[c]))
pd.DataFrame(rows).to_csv(TAB / "table_selection_model.csv", index=False)
print("=== Predictors of a non-assessable worst first-day verbal GCS (logistic OR) ===")
for r in rows:
    print("  %-16s OR %.2f (%.2f-%.2f) p=%s" % (r["predictor"], r["OR"], r["lo"], r["hi"],
          "<0.001" if r["p"] < 0.001 else "%.3f" % r["p"]))

# ---------- (A2) SMD retained vs excluded (complete-case) ----------
exc = worst[worst.na == 1]; ret = worst[worst.na == 0]
def smd(col, binary=True):
    a, b = exc[col].astype(float), ret[col].astype(float)
    if binary:
        p1, p2 = a.mean(), b.mean()
        return (p1 - p2) / np.sqrt((p1*(1-p1) + p2*(1-p2)) / 2 + 1e-9)
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-9)
smds = {"Age (years)": (round(exc.age.mean(),1), round(ret.age.mean(),1), round(smd("age", False), 2)),
        "Mechanically ventilated": (round(100*exc.vent.mean(),1), round(100*ret.vent.mean(),1), round(smd("vent"),2)),
        "Sedative infusion": (round(100*exc.sed.mean(),1), round(100*ret.sed.mean(),1), round(smd("sed"),2)),
        "Vasopressor infusion": (round(100*exc.vaso.mean(),1), round(100*ret.vaso.mean(),1), round(smd("vaso"),2)),
        "Hospital mortality": (round(100*exc.died.mean(),1), round(100*ret.died.mean(),1), round(smd("died"),2))}
pd.DataFrame([dict(variable=k, excluded=v[0], retained=v[1], smd=v[2]) for k,v in smds.items()]).to_csv(TAB/"table_smd.csv", index=False)
print("\n=== Standardized mean differences: complete-case EXCLUDED (n=%d) vs RETAINED (n=%d) ===" % (len(exc), len(ret)))
print("  %-26s %8s %8s %6s" % ("variable","excluded","retained","SMD"))
for k,v in smds.items():
    print("  %-26s %8s %8s %6.2f" % (k, v[0], v[1], v[2]))

# ---------- (B) time-course: ventilated patients with NO assessable verbal in first 24/48/72h ----------
cohort = pd.read_parquet(INT/"cohort.parquet"); cohort=cohort[cohort.first_stay]
neuro = cohort[cohort.phenotype != "comparator"]
icu = pd.read_csv(ICU/"icustays.csv.gz", usecols=["stay_id","intime"], parse_dates=["intime"])
chart = pd.read_parquet(INT/"chart_neuro.parquet")
v = chart[chart.itemid == IT["gcs_verbal"]].merge(icu, on="stay_id")
v = v[v.stay_id.isin(neuro.stay_id)]
v["hr"] = (v.charttime - v.intime).dt.total_seconds()/3600
v["assessable"] = ~v.value.eq("No Response-ETT")
ventset = set(vent.stay_id)
print("\n=== Ventilated neuro patients with NO assessable verbal GCS in the window ===")
rows=[]
for h in (24,48,72):
    w = v[(v.hr>=0)&(v.hr<=h)]
    per = w.groupby("stay_id")["assessable"].any()         # has >=1 assessable in window
    vented = per[per.index.isin(ventset)]
    pct = 100*(~vented).mean()
    rows.append(dict(window_h=h, n_ventilated=len(vented), pct_no_assessable=round(pct,1)))
    print("  first %d h: %.1f%% of ventilated patients had NO assessable verbal GCS (n=%d)" % (h, pct, len(vented)))
pd.DataFrame(rows).to_csv(TAB/"table_no_assessable_timecourse.csv", index=False)
