"""Mortality-model audit under five examination-selection frames, one code path, one bootstrap.

Frames: lowest_em, earliest, latest and carryforward_lowest_em on the corrected examination table, plus
legacy_population_lowest_em, the submitted population, kept only to reconcile with the frozen submitted table.
The SOFA cross-tabulation and the APACHE II points are a separate comparison and do not use the selected
examination: see the comment above them."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from config import REV
from lib_audit import (add_stay_covariates, add_totals, boot_auc_ci, boot_auc_diff, fit_oof, load_first_day,
                       select_exam, selected_frames)
from lib_feasible import sofa_cns

official = pd.read_parquet(REV / "official_first_day_gcs.parquet")[["stay_id", "gcs_min"]]
d1 = load_first_day(24)
STRATS = [("reconstructed", "Default-to-15 (reconstructed rule)", "tg_derived"),
          ("impute1", "Impute verbal = 1", "tg_impute1"),
          ("gcs_t", "Eye plus motor sum (T)", "tg_gcs_t"),
          ("brennan", "Brennan estimate", "tg_brennan"),
          ("complete_case", "Complete-case", "tg_drop")]
rows, out = [], {}
for rule, sel in selected_frames(24).items():
    w = add_totals(add_stay_covariates(sel))
    w = w.merge(official, on="stay_id", how="left").rename(columns={"gcs_min": "tg_official"})
    with_official = rule in ("lowest_em", "carryforward_lowest_em")
    strats = ([("official", "MIMIC first_day_gcs (official)", "tg_official")] if with_official else []) + STRATS
    preds = {}
    for slug, label, col in strats:
        ids, y, p = fit_oof(w, col)
        lo, hi = boot_auc_ci(y, p)
        rows.append(dict(rule=rule, slug=slug, strategy=label, n=len(y), deaths=int(y.sum()),
                         mortality_pct=100 * y.mean(), auroc=roc_auc_score(y, p), lo=lo, hi=hi,
                         brier=brier_score_loss(y, p), mean_gcs=w.set_index("stay_id").loc[ids, col].mean()))
        preds[slug] = pd.Series(p, index=ids)
    if rule == "lowest_em":
        base = w.set_index("stay_id")[["y", "nonassess"]]
        oof = base.join(pd.DataFrame({f"p_{k}": v for k, v in preds.items()}))
        oof.to_parquet(REV / "audit_oof_lowest_em.parquet")
        # paired AUROC differences on the same stays, same out-of-fold predictions
        for a_, b_ in [("brennan", "official"), ("brennan", "impute1"), ("brennan", "gcs_t"), ("impute1", "gcs_t")]:
            assert set(preds[a_].index) == set(preds[b_].index), f"stay sets differ for {a_} and {b_}"
            ids_ = preds[a_].index
            m_, lo_, hi_ = boot_auc_diff(base.loc[ids_, "y"].values.astype(int),
                                         preds[a_].loc[ids_].values, preds[b_].loc[ids_].values)
            out.update({f"paired_{a_}_minus_{b_}_mean": m_, f"paired_{a_}_minus_{b_}_lo": lo_,
                        f"paired_{a_}_minus_{b_}_hi": hi_, f"paired_{a_}_minus_{b_}_n": int(len(ids_))})
        # share of patients who change predicted-risk tertile between fixed imputation and Brennan
        sh = preds["impute1"].index.intersection(preds["brennan"].index)
        ta = pd.qcut(preds["impute1"].loc[sh], 3, labels=False)
        tb = pd.qcut(preds["brennan"].loc[sh], 3, labels=False)
        out["reclass_impute1_vs_brennan_pct"] = float(100 * (ta.values != tb.values).mean())
        # selection effect of complete-case handling in this population
        nax = w["nonassess"].astype(bool)
        excl, kept = w[nax], w[~nax]
        de, dk = int(excl.y.sum()), int(kept.y.sum())
        pe, pk = de / len(excl), dk / len(kept)
        rr = pe / pk
        se_rr = np.sqrt((1 - pe) / de + (1 - pk) / dk)
        se_rd = np.sqrt(pe * (1 - pe) / len(excl) + pk * (1 - pk) / len(kept))
        out.update(sel_n_total=len(w), sel_n_excluded=len(excl), sel_pct_excluded=100 * len(excl) / len(w),
                   sel_deaths_total=de + dk, sel_deaths_excluded=de, sel_pct_deaths_excluded=100 * de / (de + dk),
                   sel_mortality_excluded_pct=100 * pe, sel_mortality_retained_pct=100 * pk,
                   sel_rr=rr, sel_rr_lo=float(np.exp(np.log(rr) - 1.96 * se_rr)),
                   sel_rr_hi=float(np.exp(np.log(rr) + 1.96 * se_rr)), sel_rd_pp=100 * (pe - pk),
                   sel_rd_lo=100 * (pe - pk - 1.96 * se_rd), sel_rd_hi=100 * (pe - pk + 1.96 * se_rd))
        # SOFA CNS cross-tabulation and APACHE II GCS points (15 minus GCS). Compared quantity, on both sides:
        # the per-stay first-day MINIMUM total GCS over all examinations in the first 24 hours of the corrected
        # population. Brennan side: minimum over examinations of the Brennan total (add_totals on load_first_day(24)).
        # Official side: MIMIC first_day_gcs.gcs_min, the day minimum of per-examination totals with the default of 15
        # for non-assessable entries. It is deliberately not the total at the single lowest eye-plus-motor
        # examination used by the mortality model, which is a different quantity.
        day_min = add_totals(d1).groupby("stay_id")["tg_brennan"].min().rename("brennan_min").to_frame()
        day_min = day_min.join(official.set_index("stay_id")["gcs_min"].rename("official_min"), how="inner")
        day_min = day_min.dropna()
        out.update(sofa_n=int(len(day_min)),
                   apache_pts_official=float((15 - day_min["official_min"]).mean()),
                   apache_pts_brennan=float((15 - day_min["brennan_min"]).mean()))
        ct = pd.crosstab(day_min["brennan_min"].map(sofa_cns), day_min["official_min"].map(sofa_cns))
        ct = ct.reindex(index=range(5), columns=range(5), fill_value=0)
        ct.index.name = "brennan_estimate"
        ct.columns = [f"official_{c}" for c in ct.columns]
        ct.to_csv(REV / "sofa_crosstab.csv")
        # cells quoted in the Results: stays with an official first-day SOFA CNS score of 0, and how many of them
        # score 3 or 4 when the first-day minimum is computed with the Brennan estimate
        out.update(sofa_official0_n=int(ct["official_0"].sum()),
                   sofa_brennan3_official0_n=int(ct.loc[3, "official_0"]),
                   sofa_brennan4_official0_n=int(ct.loc[4, "official_0"]))

res = pd.DataFrame(rows)
res.to_csv(REV / "audit_by_rule.csv", index=False)
for r in res.itertuples():
    for k in ("n", "deaths", "mortality_pct", "auroc", "lo", "hi", "brier", "mean_gcs"):
        out[f"{r.slug}_{r.rule}_{k}"] = float(getattr(r, k))
(REV / "audit.json").write_text(json.dumps(out, indent=1))
print(res.round(4).to_string(index=False))

# per-phenotype rows for eTable 3 (primary rule), same code path and seeds
w = add_totals(add_stay_covariates(select_exam(d1, "lowest_em")))
pr = []
for ph, wp in w.groupby("phenotype"):
    for slug, label, col in STRATS[1:]:
        ids, y, p = fit_oof(wp, col)
        lo, hi = boot_auc_ci(y, p)
        pr.append(dict(phenotype=ph, slug=slug, strategy=label, n=len(y), deaths=int(y.sum()),
                       auroc=roc_auc_score(y, p), lo=lo, hi=hi, brier=brier_score_loss(y, p),
                       mean_gcs=wp.set_index("stay_id").loc[ids, col].mean()))
pd.DataFrame(pr).to_csv(REV / "audit_by_phenotype.csv", index=False)
