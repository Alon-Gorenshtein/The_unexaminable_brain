"""The main estimates repeated with diagnosis codes specific to each phenotype, and with treatment covariates
limited to the first 24 hours.

The primary definition labels a stay by any code in a block: 432 and I62 for SDH, 433-434 and I63 for AIS,
800-804, 850-854 and S06 for TBI, 3481, 3485, G931 and G9382 for anoxic. The specific definition keeps the codes
specific to the labeled injury (config.specific_phenotype_masks); for TBI these are ICD-9 850 to 854 and ICD-9 800,
801, 803 and 804 only with a fourth digit other than 0 or 5, so facial fractures and skull fractures without mention
of intracranial injury no longer count. A stay with none of the specific codes is dropped. One function computes
every estimate for a set of labelled stays; it runs on all eligible stays and on those with a specific label, so both
columns share one code path. The primary column is checked against the registered outputs before anything is
written. The first-day refit replaces the any-time-in-stay vasopressor and ventilation covariates (which count
exposure after the examination window) with exposure that starts within 24 h of ICU admission, on the primary
audit stays."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import HOSP, INT, REV
from lib_audit import (COVARS, add_stay_covariates, add_totals, boot_auc_diff, exposed_by, fit_oof, load_first_day,
                       pct_exposed_after_exam, select_exam)
from lib_cohort import NEURO, specific_labels
from lib_feasible import outside_total_bounds

STRATS = {"official": "tg_official", "reconstructed": "tg_derived", "impute1": "tg_impute1", "gcs_t": "tg_gcs_t",
          "brennan": "tg_brennan"}

c = pd.read_parquet(INT / "cohort.parquet")
neuro = c[c.first_stay & (c.phenotype != "comparator")].set_index("stay_id")
burden = pd.read_parquet(INT / "burden_per_stay.parquet")
official = pd.read_parquet(REV / "official_first_day_gcs.parquet")
vent_stays = set(pd.read_parquet(INT / "vent.parquet", columns=["stay_id"]).stay_id)     # as in Table 1
audit_frame = add_totals(add_stay_covariates(select_exam(load_first_day(24), "lowest_em")))
audit_frame = audit_frame.merge(official[["stay_id", "gcs_min"]], on="stay_id", how="left")
audit_frame = audit_frame.rename(columns={"gcs_min": "tg_official"})


def fit_strategies(w, covars):
    """Out-of-fold predictions of the five strategies: stay ids, outcome, {strategy: prediction}."""
    preds = {}
    for slug, col in STRATS.items():
        sid, y, preds[slug] = fit_oof(w, col, covars)
        assert len(sid) == len(w), f"{slug}: stays dropped from the audit"
    return sid, y, preds


def estimates(labels):
    """Every estimate for the stays in `labels` (Series, stay_id -> phenotype)."""
    ids = labels.index
    out = {"eligible_n": len(ids)}
    b = burden[burden.stay_id.isin(ids)].assign(ph=lambda d: d.stay_id.map(labels))
    out.update(analytic_n=len(b), ever_pct=100 * b.ever_nonassess.mean())
    for ph in NEURO:
        g = b[b.ph == ph]
        out.update({f"ever_{ph}_n": len(g), f"ever_{ph}_pct": 100 * g.ever_nonassess.mean(),
                    f"vent_frac_{ph}_pct": 100 * g[g.ventilated].frac_nonassess.median()})
    # Table 1 characteristics of the anoxic stays, over the stays with a verbal entry (the Table 1 cohort)
    an = pd.Index(b.loc[b.ph == "anoxic", "stay_id"])
    out.update(mortality_anoxic_pct=100 * (neuro.loc[an, "hospital_expire_flag"] == 1).mean(),
               vent_anoxic_pct=100 * an.isin(vent_stays).mean())
    # official first-day GCS of 15, and the 15s the default rule assigned that eye and motor make impossible
    o = official[official.stay_id.isin(ids) & official.gcs_min.notna()]
    o15 = o[o.gcs_min == 15]
    sel = o15[(o15.gcs_unable == 1) | (o15.gcs_verbal == 0)]
    em = sel.gcs_eyes.fillna(4) + sel.gcs_motor.fillna(6)       # gcs.sql defaults for a missing component
    out.update(official_n=len(o), official_15_pct=100 * len(o15) / len(o), default_15_n=len(sel),
               default_15_impossible_pct=100 * np.asarray(outside_total_bounds(em.values, 15)).mean())
    # mortality audit on the lowest eye-plus-motor examination of the first 24 h
    w = audit_frame[audit_frame.stay_id.isin(ids)]
    na = w.nonassess.astype(bool)
    out.update(audit_n=len(w), audit_deaths=int(w.y.sum()), nonassess_n=int(na.sum()),
               cc_pct_deaths_excluded=100 * w.y[na].sum() / w.y.sum())
    sid, y, preds = fit_strategies(w, COVARS)
    out.update({f"auroc_{slug}": roc_auc_score(y, p) for slug, p in preds.items()})
    out["diff_mean"], out["diff_lo"], out["diff_hi"] = boot_auc_diff(y, preds["brennan"], preds["official"])
    nas = w.set_index("stay_id").loc[sid, "nonassess"].values.astype(bool)     # fit_oof returns stays sorted by id
    for slug in ("official", "brennan"):
        out[f"oe_nonassess_{slug}"] = y[nas].mean() / preds[slug][nas].mean()
    return out


def firstday_estimates(w):
    """The audit for the stays in `w` with vasopressor and ventilation exposure limited to the first 24 h."""
    intime = c.set_index("stay_id").intime
    inf = pd.read_parquet(INT / "infusions.parquet", columns=["stay_id", "starttime", "cls"])
    vent = pd.read_parquet(INT / "vent.parquet", columns=["stay_id", "starttime"])
    w = w.assign(vaso_firstday=w.stay_id.isin(exposed_by(inf[inf.cls == "vasopressor"], intime)).astype(int),
                 vent_firstday=w.stay_id.isin(exposed_by(vent, intime)).astype(int))
    sid, y, preds = fit_strategies(w, ("age", "vaso_firstday", "vent_firstday"))
    out = {f"firstday_auroc_{slug}": roc_auc_score(y, p) for slug, p in preds.items()}
    out["firstday_diff_mean"], out["firstday_diff_lo"], out["firstday_diff_hi"] = \
        boot_auc_diff(y, preds["brennan"], preds["official"])
    out.update(firstday_vaso_pct=100 * w.vaso_firstday.mean(), firstday_vent_pct=100 * w.vent_firstday.mean(),
               stay_vaso_pct=100 * w.vaso_stay.mean(), stay_vent_pct=100 * w.vent_stay.mean())
    # a first-day exposure can still start after the selected examination
    exam_time = w.set_index("stay_id").charttime
    out.update(firstday_vaso_after_exam_pct=pct_exposed_after_exam(inf[inf.cls == "vasopressor"], intime, exam_time),
               firstday_vent_after_exam_pct=pct_exposed_after_exam(vent, intime, exam_time))
    return out


def check_primary(p):
    """The primary column must reproduce the registered outputs; a drift stops the run."""
    f, a, acc, cmp_ = (json.loads((REV / n).read_text())
                       for n in ("feasible.json", "audit.json", "cohort_accounting.json", "comparability.json"))
    exact = dict(eligible_n=acc["neuro_first_stay_n"], analytic_n=acc["with_verbal_n"], official_n=f["official_n"],
                 default_15_n=f["official_default_15_n"], audit_n=a["sel_n_total"], audit_deaths=a["sel_deaths_total"],
                 nonassess_n=a["sel_n_excluded"])
    close = dict(ever_pct=100 * acc["ever_nonassess_n"] / acc["with_verbal_n"], official_15_pct=f["official_15_pct"],
                 default_15_impossible_pct=f["official_default_15_impossible_pct"],
                 cc_pct_deaths_excluded=a["sel_pct_deaths_excluded"], diff_mean=a["paired_brennan_minus_official_mean"],
                 diff_lo=a["paired_brennan_minus_official_lo"], diff_hi=a["paired_brennan_minus_official_hi"],
                 oe_nonassess_official=cmp_["nonassessable_official_oe_ratio"],
                 oe_nonassess_brennan=cmp_["nonassessable_brennan_oe_ratio"],
                 **{f"auroc_{s}": a[f"{s}_lowest_em_auroc"] for s in STRATS})
    for k, v in exact.items():
        assert p[k] == v, f"primary_{k}: {p[k]} != registered {v}"
    for k, v in close.items():
        assert abs(p[k] - v) < 5e-4, f"primary_{k}: {p[k]} != registered {v}"


dx = pd.read_csv(HOSP / "diagnoses_icd.csv.gz", usecols=["hadm_id", "icd_code", "icd_version"], dtype={"icd_code": str})
dx["icd_code"] = dx.icd_code.str.strip().str.upper()
dx = dx[dx.hadm_id.isin(neuro.hadm_id)]
primary = neuro.phenotype
specific = neuro.hadm_id.map(specific_labels(dx, neuro.hadm_id.unique())).dropna()

est = {"primary": estimates(primary), "specific": estimates(specific)}
check_primary(est["primary"])
out = {f"{col}_{k}": v for col, e in est.items() for k, v in e.items()}

# the stays the specific definition drops, grouped by their primary label
removed = primary[~primary.index.isin(specific.index)]
assert set(removed) <= {"TBI", "AIS", "anoxic", "SDH"}, "SAH and ICH keep their codes, so none can be removed"
out.update(removed_n=len(removed), removed_pct=100 * len(removed) / len(primary),
           **{f"removed_{ph}_n": int((removed == ph).sum()) for ph in ("TBI", "AIS", "anoxic", "SDH")},
           removed_mortality_pct=100 * neuro.loc[removed.index, "hospital_expire_flag"].mean())
# the stays the specific definition drops, by the ICD version of their diagnosis codes (one version per admission)
assert (dx.groupby("hadm_id").icd_version.nunique() == 1).all(), "an admission carries codes of two ICD versions"
version = neuro.hadm_id.map(dx.groupby("hadm_id").icd_version.first())
out.update(removed_icd9_n=int((version[removed.index] == 9).sum()),
           removed_icd10_n=int((version[removed.index] == 10).sum()),
           removed_icd9_pct=100 * (version[removed.index] == 9).sum() / (version == 9).sum(),
           removed_icd10_pct=100 * (version[removed.index] == 10).sum() / (version == 10).sum())
# removed stays are not added to the comparator above; this is its percentage if they were
comp = burden[burden.phenotype == "comparator"]
moved = pd.concat([comp, burden[burden.stay_id.isin(removed.index)]])
out.update(comparator_ever_pct=100 * comp.ever_nonassess.mean(),
           comparator_ever_moved_pct=100 * moved.ever_nonassess.mean())
out.update(firstday_estimates(audit_frame[audit_frame.stay_id.isin(primary.index)]))
text = json.dumps(out, indent=1, allow_nan=False)
(REV / "definition_sensitivity.json").write_text(text)
print(text)
