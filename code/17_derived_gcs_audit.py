"""HEADLINE ANALYSIS: raw component-aware GCS vs the official MIMIC-derived GCS.
The MIMIC-code gcs.sql recodes 'No Response-ETT' -> verbal=0, then sets the TOTAL GCS to 15
('The GCS for sedated patients is defaulted to 15'). We quantify how often the derived first-day
GCS overstates a component-aware GCS, the severity miscategorization (SOFA CNS / APACHE GCS points),
and the effect on the mortality model."""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from config import INT, TAB
from lib_gcs_strategies import _kramer_add
RNG = np.random.RandomState(0)


def sofa_cns(g):
    if g >= 15: return 0
    if g >= 13: return 1
    if g >= 10: return 2
    if g >= 6:  return 3
    return 4


ex = pd.read_parquet(INT / "aim4_exams.parquet")          # first 72h exams, neuro first-stay
d1 = ex[ex.hr <= 24].dropna(subset=["eye", "motor"]).copy()
d1["em"] = d1.eye + d1.motor
# per-exam GCS under the two constructions
d1["derived"] = np.where(d1.nonassess, 15, d1.em + d1.verbal_score)       # MIMIC gcs.sql rule
d1["compaware"] = np.where(d1.nonassess, d1.em + d1.em.map(_kramer_add),  # component-aware (Brennan)
                           d1.em + d1.verbal_score)
# first-day MINIMUM per patient (mirrors MIMIC first_day_gcs, which takes the daily min)
g = d1.groupby("stay_id").agg(derived_min=("derived", "min"),
                              compaware_min=("compaware", "min"),
                              any_ett=("nonassess", "any"),
                              em_min=("em", "min"),
                              y=("y", "first")).reset_index()
g["gap"] = g.derived_min - g.compaware_min
g["sofa_derived"] = g.derived_min.map(sofa_cns)
g["sofa_compaware"] = g.compaware_min.map(sofa_cns)
g["apache_gcs_pts_derived"] = 15 - g.derived_min        # APACHE II GCS contribution
g["apache_gcs_pts_compaware"] = 15 - g.compaware_min

n = len(g)
print("=== Raw component-aware vs MIMIC-derived first-day GCS (n=%d neuro first-stay) ===" % n)
print("Patients with >=1 non-assessable (ETT) first-day exam:", int(g.any_ett.sum()),
      "(%.1f%%)" % (100 * g.any_ett.mean()))
print("Derived first-day GCS = 15 (normal):", int((g.derived_min == 15).sum()),
      "(%.1f%%)" % (100 * (g.derived_min == 15).mean()))
over = g[g.gap > 0]
print("Patients whose derived GCS OVERSTATES the component-aware GCS:", len(over),
      "(%.1f%% of cohort)" % (100 * len(over) / n))
print("  among them, median overstatement: %.0f points (IQR %.0f-%.0f)"
      % (over.gap.median(), over.gap.quantile(.25), over.gap.quantile(.75)))
print("  derived=15 but component-aware <=8 (moderate-severe):",
      int(((g.derived_min == 15) & (g.compaware_min <= 8)).sum()),
      "(%.1f%% of cohort)" % (100 * ((g.derived_min == 15) & (g.compaware_min <= 8)).mean()))
# SOFA CNS miscategorization
mis = g[(g.sofa_derived == 0) & (g.sofa_compaware >= 3)]
print("Derived SOFA CNS = 0 (normal) but component-aware SOFA CNS >= 3 (GCS<=9):",
      len(mis), "(%.1f%% of cohort)" % (100 * len(mis) / n))
print("Mean APACHE II GCS points: derived %.2f vs component-aware %.2f (lower derived = less severe)"
      % (g.apache_gcs_pts_derived.mean(), g.apache_gcs_pts_compaware.mean()))
# save summary
summ = dict(n=n, pct_any_ett=round(100*g.any_ett.mean(),1),
            pct_derived15=round(100*(g.derived_min==15).mean(),1),
            n_overstated=len(over), pct_overstated=round(100*len(over)/n,1),
            median_overstatement=float(over.gap.median()),
            pct_derived15_compaware_le8=round(100*((g.derived_min==15)&(g.compaware_min<=8)).mean(),1),
            n_sofa_miscat=len(mis), pct_sofa_miscat=round(100*len(mis)/n,1),
            apache_pts_derived=round(g.apache_gcs_pts_derived.mean(),2),
            apache_pts_compaware=round(g.apache_gcs_pts_compaware.mean(),2))
pd.Series(summ).to_csv(TAB/"table_derived_audit.csv")
# SOFA category cross-tab
ct = pd.crosstab(g.sofa_compaware, g.sofa_derived, rownames=["component-aware"], colnames=["MIMIC-derived"])
ct.to_csv(TAB/"table_sofa_crosstab.csv"); print("\nSOFA CNS cross-tab (rows=component-aware, cols=derived):\n", ct.to_string())

# ---- mortality model: add MIMIC-derived as a strategy (worst-exam, consistent with Aim 3) ----
worst = d1.sort_values(["stay_id","em","charttime"]).groupby("stay_id").first().reset_index()
inf=pd.read_parquet(INT/"infusions.parquet"); vent=pd.read_parquet(INT/"vent.parquet")
# age and y (hospital_expire_flag) already present from the exam table
worst["vaso"]=worst.stay_id.isin(inf[inf.cls=="vasopressor"].stay_id).astype(int)
worst["vent"]=worst.stay_id.isin(vent.stay_id).astype(int)
worst["y"]=worst["y"].astype(int)
worst["tg_derived"]=np.where(worst.nonassess,15,worst.em+worst.verbal_score)
worst["tg_brennan"]=np.where(worst.nonassess,worst.em+worst.em.map(_kramer_add),worst.em+worst.verbal_score)
def auroc(col):
    d=worst.dropna(subset=[col]); feats=d[[col,"age","vaso","vent"]].astype(float).values; yy=d.y.values
    oof=np.zeros(len(yy))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(feats,yy):
        oof[te]=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(feats[tr],yy[tr]).predict_proba(feats[te])[:,1]
    idx=np.arange(len(yy)); aucs=[]
    for _ in range(500):
        s=RNG.choice(idx,len(yy),replace=True)
        if yy[s].sum() not in (0,len(s)): aucs.append(roc_auc_score(yy[s],oof[s]))
    return round(roc_auc_score(yy,oof),3), round(np.percentile(aucs,2.5),3), round(np.percentile(aucs,97.5),3), round(float(d[col].mean()),2)
print("\n=== Mortality model: MIMIC-derived vs component-aware (Brennan) worst-exam GCS ===")
for name,col in [("MIMIC-derived (=15 if ETT)","tg_derived"),("Component-aware (Brennan)","tg_brennan")]:
    a,lo,hi,mt=auroc(col); print("  %-28s AUROC %.3f (%.3f-%.3f), mean GCS %.2f" % (name,a,lo,hi,mt))
