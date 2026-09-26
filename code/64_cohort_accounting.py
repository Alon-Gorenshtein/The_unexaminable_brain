"""Who was excluded for having no verbal GCS, and who fell out of the audit subset.
Also: diagnostic overlap before the hierarchy, and results under four alternative rules."""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import HOSP, INT, PHENOTYPE_PRIORITY, REV
from lib_audit import load_first_day
from lib_cohort import NEURO, assign_hier, phenotype_flags, primary_dx_label

c = pd.read_parquet(INT / "cohort.parquet")
neuro = c[c.first_stay & (c.phenotype != "comparator")].copy()
burden = pd.read_parquet(INT / "burden_per_stay.parquet")
vent = pd.read_parquet(INT / "vent.parquet")
neuro["vent"] = neuro.stay_id.isin(vent.stay_id)
neuro["has_verbal"] = neuro.stay_id.isin(burden.loc[burden.phenotype != "comparator", "stay_id"])
neuro["in_aim3"] = neuro.stay_id.isin(set(load_first_day(24).stay_id))
neuro["in_aim3_legacy"] = neuro.stay_id.isin(set(load_first_day(24, legacy=True).stay_id))


def summ(df):
    return {"n": len(df), "age_median": df.age.median(), "age_q1": df.age.quantile(.25),
            "age_q3": df.age.quantile(.75), "female_pct": 100 * (df.gender == "F").mean(),
            "ventilated_pct": 100 * df.vent.mean(), "mortality_pct": 100 * df.hospital_expire_flag.mean(),
            "icu_los_median_days": df.los.median()}


rows = []
for name, flag, sub in [("verbal_documented", "has_verbal", neuro), ("audit_subset", "in_aim3", neuro[neuro.has_verbal])]:
    for label, g in [("included", sub[sub[flag]]), ("excluded", sub[~sub[flag]])]:
        rows.append(dict(comparison=name, group=label, **summ(g)))
pd.DataFrame(rows).to_csv(REV / "excluded_vs_included.csv", index=False)

out = dict(neuro_first_stay_n=int(len(neuro)), with_verbal_n=int(neuro.has_verbal.sum()),
           no_verbal_n=int((~neuro.has_verbal).sum()), aim3_n=int(neuro.in_aim3.sum()),
           aim3_excluded_n=int((neuro.has_verbal & ~neuro.in_aim3).sum()))
out["no_verbal_pct"] = float(100 * out["no_verbal_n"] / out["neuro_first_stay_n"])
out["aim3_legacy_n"] = int(neuro.in_aim3_legacy.sum())
# what the submitted analysis dropped: patients with a verbal entry outside its audit subset
out["aim3_legacy_excluded_n"] = out["with_verbal_n"] - out["aim3_legacy_n"]

# patients with a verbal entry who are still outside the corrected audit subset
# (no complete same-time eye, motor and verbal examination in 0-24 h)
off = pd.read_parquet(REV / "official_first_day_gcs.parquet").set_index("stay_id").gcs_min
lost = neuro[neuro.has_verbal & ~neuro.in_aim3].copy()
lost["hr_first_verbal"] = (lost.stay_id.map(burden.set_index("stay_id")["first_time"]) - lost.intime).dt.total_seconds() / 3600
dt = pd.to_datetime(lost.deathtime)                      # stored as text in cohort.parquet
lost["died_24h"] = dt.notna() & ((dt - lost.intime).dt.total_seconds() / 3600 <= 24)
out.update(aim3_excl_first_verbal_after24h_n=int((lost.hr_first_verbal > 24).sum()),
           aim3_excl_verbal_le24h_no_triple_n=int((lost.hr_first_verbal <= 24).sum()),
           aim3_excl_died_le24h_n=int(lost.died_24h.sum()),
           aim3_excl_official_gcs_available_n=int(lost.stay_id.map(off).notna().sum()))

# ---- diagnostic overlap ----
dx = pd.read_csv(HOSP / "diagnoses_icd.csv.gz", usecols=["hadm_id", "seq_num", "icd_code", "icd_version"],
                 dtype={"icd_code": str})
dx["icd_code"] = dx.icd_code.str.strip().str.upper()
ids = neuro.hadm_id.unique()
dx = dx[dx.hadm_id.isin(ids)]
flags = phenotype_flags(dx, ids)
k = flags.sum(axis=1)
out.update(overlap_total_n=int(len(flags)), overlap_1_n=int((k == 1).sum()), overlap_2_n=int((k == 2).sum()),
           overlap_3plus_n=int((k >= 3).sum()), overlap_multi_pct=float(100 * (k >= 2).mean()))
pairs = flags.astype(int).T.dot(flags.astype(int))
pairs.to_csv(REV / "phenotype_overlap_pairs.csv")

# ---- alternative hierarchy rules ----
labels = {"primary": assign_hier(flags, PHENOTYPE_PRIORITY),
          "reversed": assign_hier(flags, PHENOTYPE_PRIORITY[::-1]),
          "single_only": assign_hier(flags, PHENOTYPE_PRIORITY).where(k == 1),
          "first_listed_dx": primary_dx_label(dx, ids)}
b = burden[burden.phenotype != "comparator"].merge(neuro[["stay_id", "hadm_id"]], on="stay_id")


def per_phenotype(df, col):
    r = []
    for ph, g in df.dropna(subset=[col]).groupby(col):
        v = g[g.ventilated]
        r.append(dict(phenotype=ph, n=len(g), ever_na_pct=100 * g.ever_nonassess.mean(),
                      vent_median_frac_pct=100 * v.frac_nonassess.median() if len(v) else np.nan))
    return pd.DataFrame(r)


parts = []
for rule, lab in labels.items():
    d = b.assign(lab=b.hadm_id.map(lab))
    parts.append(per_phenotype(d, "lab").assign(rule=rule))
long = pd.concat([b.assign(lab=ph)[flags.loc[b.hadm_id, ph].values] for ph in NEURO])
parts.append(per_phenotype(long, "lab").assign(rule="any_code"))
h = pd.concat(parts)
h.to_csv(REV / "hierarchy_sensitivity.csv", index=False)
prim = h[h.rule == "primary"].set_index("phenotype").ever_na_pct
for rule in ("reversed", "single_only", "first_listed_dx", "any_code"):
    o = h[h.rule == rule].set_index("phenotype").ever_na_pct.reindex(prim.index)
    out[f"rank_corr_{rule}"] = float(spearmanr(prim, o)[0])
# every count is a Python int and every percentage a float; a NaN anywhere must stop the run
(REV / "cohort_accounting.json").write_text(json.dumps(out, indent=1, allow_nan=False))
print(json.dumps(out, indent=1, allow_nan=False))
