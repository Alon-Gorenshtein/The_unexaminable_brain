"""Comprehensive inferential statistics: formal tests, effect sizes, 95% CIs,
Benjamini-Hochberg correction across the primary family."""
import warnings, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, cohen_kappa_score, accuracy_score
from config import INT, TAB, ICU, IT, EYE_SCORE, MOTOR_SCORE
from lib_gcs_strategies import total_gcs, _kramer_add

RNG = np.random.RandomState(2026)
NEURO = ["TBI", "ICH", "SAH", "SDH", "AIS", "anoxic"]
out = {}


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan)
    lo, hi = sm.stats.proportion_confint(k, n, method="wilson")
    return round(100 * lo, 1), round(100 * hi, 1)


def cramers_v(ct):
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.sum().sum()
    r, k = ct.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def epsilon_sq(H, n, k):
    return (H - k + 1) / (n - k)


# ---------------- Aim 1: burden ----------------
b = pd.read_parquet(INT / "burden_per_stay.parquet")
bn = b[b.phenotype != "comparator"]
# chi-square: ever non-assessable x phenotype (6 neuro groups)
ct = pd.crosstab(bn.phenotype, bn.ever_nonassess)
chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
out["aim1_ever_chi2"] = dict(chi2=round(chi2, 1), df=int(dof), p=p_chi,
                             cramers_v=round(cramers_v(ct), 3))
# Kruskal-Wallis on per-stay fraction among VENTILATED patients across phenotypes
vent = b[b.ventilated & (b.phenotype != "comparator")]
groups = [vent[vent.phenotype == ph]["frac_nonassess"].values for ph in NEURO]
H, p_kw = stats.kruskal(*groups)
out["aim1_frac_kruskal"] = dict(H=round(H, 1),
                                eps2=round(epsilon_sq(H, len(vent), len(NEURO)), 3), p=p_kw)
# adjusted logistic: ever_nonassess ~ phenotype (ref=AIS) + age + female  (vent is the mediator, excluded)
d = bn.merge(pd.read_parquet(INT / "cohort.parquet")[["stay_id", "gender", "age"]], on="stay_id")
d["female"] = (d.gender == "F").astype(int)
dd = d.dropna(subset=["age"]).copy()
dum = pd.get_dummies(dd["phenotype"], prefix="ph")
ref = "ph_AIS"
Xcols = [c for c in dum.columns if c != ref]
Xmat = sm.add_constant(pd.concat([dum[Xcols], dd[["age", "female"]].reset_index(drop=True)], axis=1).astype(float))
ymat = dd["ever_nonassess"].astype(int).values
res = sm.Logit(ymat, Xmat).fit(disp=0)
ors = []
for c in Xcols:
    ors.append(dict(term=c.replace("ph_", "") + " vs AIS", OR=round(np.exp(res.params[c]), 2),
                    lo=round(np.exp(res.conf_int().loc[c, 0]), 2),
                    hi=round(np.exp(res.conf_int().loc[c, 1]), 2), p=res.pvalues[c]))
out["aim1_adj_or"] = ors

# ---------------- Aim 2: timing + concordance ----------------
dark = pd.read_parquet(INT / "darktimes.parquet")
dn = dark[dark.phenotype.isin(NEURO)].dropna(subset=["t_first_dark"])
gg = [dn[dn.phenotype == ph]["t_first_dark"].values for ph in NEURO]
H2, p2 = stats.kruskal(*gg)
out["aim2_timing_kruskal"] = dict(H=round(H2, 1),
                                  eps2=round(epsilon_sq(H2, len(dn), len(NEURO)), 3), p=p2)
conc = pd.read_csv(TAB / "table_concordance.csv", index_col=0)
out["aim2_vent_concordance_range"] = [round(conc.loc[NEURO, "vent"].min(), 1),
                                      round(conc.loc[NEURO, "vent"].max(), 1)]

# ---------------- Aim 3: selection effect (RR, RD) ----------------
sel = pd.read_csv(TAB / "table_selection_effect.csv", index_col=0)["0"]
ne, nr = int(sel["n_excluded"]), int(sel["n_total"] - sel["n_excluded"])
de = int(sel["deaths_excluded"]); dr = int(sel["deaths_total"] - sel["deaths_excluded"])
pe, pr = de / ne, dr / nr
rr = pe / pr
se_log_rr = np.sqrt((1 - pe) / de + (1 - pr) / dr)
rr_ci = (np.exp(np.log(rr) - 1.96 * se_log_rr), np.exp(np.log(rr) + 1.96 * se_log_rr))
rd = pe - pr
se_rd = np.sqrt(pe * (1 - pe) / ne + pr * (1 - pr) / nr)
ct2 = np.array([[de, ne - de], [dr, nr - dr]])
chi_s, p_sel, _, _ = stats.chi2_contingency(ct2)
out["aim3_selection"] = dict(rr=round(rr, 2), rr_lo=round(rr_ci[0], 2), rr_hi=round(rr_ci[1], 2),
                             rd_pp=round(100 * rd, 1), rd_lo=round(100 * (rd - 1.96 * se_rd), 1),
                             rd_hi=round(100 * (rd + 1.96 * se_rd), 1), p=p_sel)

# ---------------- Aim 3: paired AUROC differences among imputation strategies ----------------
cohort = pd.read_parquet(INT / "cohort.parquet"); cohort = cohort[cohort.first_stay]
neuro = cohort[cohort.phenotype != "comparator"]
icu = pd.read_csv(ICU / "icustays.csv.gz", usecols=["stay_id", "intime"], parse_dates=["intime"])
chart = pd.read_parquet(INT / "chart_neuro.parquet").merge(icu, on="stay_id")
chart["hr"] = (chart.charttime - chart.intime).dt.total_seconds() / 3600
d1 = chart[(chart.hr >= 0) & (chart.hr <= 24)]
eye = d1[d1.itemid == IT["gcs_eye"]].assign(eye=lambda x: x["value"].map(EYE_SCORE))
mot = d1[d1.itemid == IT["gcs_motor"]].assign(motor=lambda x: x["value"].map(MOTOR_SCORE))
vrb = d1[d1.itemid == IT["gcs_verbal"]].rename(columns={"value": "verbal_raw"})
trip = (eye[["stay_id", "charttime", "eye"]].merge(mot[["stay_id", "charttime", "motor"]], on=["stay_id", "charttime"])
        .merge(vrb[["stay_id", "charttime", "verbal_raw"]], on=["stay_id", "charttime"]).dropna())
trip["em"] = trip.eye + trip.motor
worst = trip.sort_values(["stay_id", "em", "charttime"]).groupby("stay_id").first().reset_index()
X = neuro.merge(worst, on="stay_id").set_index("stay_id")
inf = pd.read_parquet(INT / "infusions.parquet"); ventp = pd.read_parquet(INT / "vent.parquet")
X["vaso"] = X.index.isin(inf[inf.cls == "vasopressor"].stay_id).astype(int)
X["vent"] = X.index.isin(ventp.stay_id).astype(int)
X["y"] = X.hospital_expire_flag.astype(int)
yv = X.y.values
preds = {}
for s in ["impute1", "gcs_t", "kramer"]:
    tg = np.array([total_gcs(dict(eye=e, motor=m, verbal_raw=v), s) for e, m, v in zip(X.eye, X.motor, X.verbal_raw)], float)
    feats = np.column_stack([tg, X.age.values, X.vaso.values, X.vent.values])
    oof = np.zeros(len(yv))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(feats, yv):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(feats[tr], yv[tr])
        oof[te] = clf.predict_proba(feats[te])[:, 1]
    preds[s] = oof


def boot_auc_diff(y, pa, pb, B=2000):
    idx = np.arange(len(y)); diffs = []
    for _ in range(B):
        s = RNG.choice(idx, len(y), replace=True)
        if y[s].sum() in (0, len(s)):
            continue
        diffs.append(roc_auc_score(y[s], pa[s]) - roc_auc_score(y[s], pb[s]))
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return round(float(np.mean(diffs)), 4), round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4), max(p, 1 / B)

cmp = []
for a, bl in [("kramer", "impute1"), ("kramer", "gcs_t"), ("impute1", "gcs_t")]:
    md, lo, hi, p = boot_auc_diff(yv, preds[a], preds[bl])
    nm = {"kramer": "Brennan"}
    cmp.append(dict(contrast=f"{nm.get(a,a)} vs {nm.get(bl,bl)}", auroc_diff=md, lo=lo, hi=hi, p=p))
out["aim3_auroc_diffs"] = cmp

# ---------------- Aim 4: learned vs Brennan QWK difference ----------------
ex = pd.read_parquet(INT / "aim4_exams.parquet")
A = ex[~ex.nonassess].dropna(subset=["verbal_score", "eye", "motor"]).copy()
A["verbal_score"] = A.verbal_score.astype(int); A["em"] = A.eye + A.motor
# reuse stored learned OOF if available else quick recompute of QWK diff via bootstrap on recovery table
rec = pd.read_csv(TAB / "table_aim4_recovery.csv")
lrn = rec[rec.method.str.startswith("Learned")].iloc[0]
brn = rec[rec.method.str.startswith("Brennan")].iloc[0]
out["aim4_recovery_gap"] = dict(qwk_learned=lrn.qwk, qwk_brennan=brn.qwk,
                                qwk_gain=round(lrn.qwk - brn.qwk, 3),
                                acc_learned=lrn.accuracy, acc_brennan=brn.accuracy,
                                ci_nonoverlap=bool(lrn.qwk_lo > brn.qwk_hi))

# ---------------- BH correction across the primary family ----------------
fam = {
    "Aim1 ever-nonassess x phenotype": out["aim1_ever_chi2"]["p"],
    "Aim1 fraction (ventilated) x phenotype": out["aim1_frac_kruskal"]["p"],
    "Aim2 time-to-dark x phenotype": out["aim2_timing_kruskal"]["p"],
    "Aim3 selection-effect mortality": out["aim3_selection"]["p"],
}
keys = list(fam); pv = [fam[k] for k in keys]
rej, padj, _, _ = multipletests(pv, alpha=0.05, method="fdr_bh")
out["bh_family"] = {k: dict(p_raw=pv[i], p_adj=float(padj[i]), sig=bool(rej[i])) for i, k in enumerate(keys)}

with open(INT.parent / "stats_digest.json", "w") as f:
    json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))

# pretty print
def fp(p): return "<0.001" if p < 0.001 else f"{p:.3f}"
print("=== AIM 1 ===")
print("Ever non-assessable x phenotype: chi2(%d)=%.1f, Cramer V=%.3f, p=%s" % (
    out["aim1_ever_chi2"]["df"], out["aim1_ever_chi2"]["chi2"], out["aim1_ever_chi2"]["cramers_v"], fp(out["aim1_ever_chi2"]["p"])))
print("Fraction (ventilated) x phenotype: Kruskal H=%.1f, eps2=%.3f, p=%s" % (
    out["aim1_frac_kruskal"]["H"], out["aim1_frac_kruskal"]["eps2"], fp(out["aim1_frac_kruskal"]["p"])))
print("Adjusted ORs (ever non-assessable, ref=AIS, adj age+sex):")
for o in out["aim1_adj_or"]:
    print("   %-16s OR %.2f (%.2f-%.2f) p=%s" % (o["term"], o["OR"], o["lo"], o["hi"], fp(o["p"])))
print("=== AIM 2 ===")
print("Time-to-dark x phenotype: Kruskal H=%.1f, eps2=%.3f, p=%s" % (
    out["aim2_timing_kruskal"]["H"], out["aim2_timing_kruskal"]["eps2"], fp(out["aim2_timing_kruskal"]["p"])))
print("=== AIM 3 ===")
s = out["aim3_selection"]
print("Selection effect mortality excluded vs retained: RR %.2f (%.2f-%.2f), RD %.1f pp (%.1f-%.1f), p=%s" % (
    s["rr"], s["rr_lo"], s["rr_hi"], s["rd_pp"], s["rd_lo"], s["rd_hi"], fp(s["p"])))
print("Paired AUROC differences (imputation strategies):")
for c in out["aim3_auroc_diffs"]:
    print("   %-22s dAUROC %+.4f (%.4f, %.4f) p=%s" % (c["contrast"], c["auroc_diff"], c["lo"], c["hi"], fp(c["p"])))
print("=== AIM 4 ===")
g = out["aim4_recovery_gap"]
print("QWK learned %.3f vs Brennan %.3f (gain %.3f, CIs non-overlapping=%s)" % (
    g["qwk_learned"], g["qwk_brennan"], g["qwk_gain"], g["ci_nonoverlap"]))
print("=== BH family ===")
for k, v in out["bh_family"].items():
    print("   %-40s p_raw=%s p_adj=%s sig=%s" % (k, fp(v["p_raw"]), fp(v["p_adj"]), v["sig"]))
print("\nWrote stats_digest.json")
