import pandas as pd, numpy as np
import json
from config import INT, TAB, REV
from lib_cohort import hospital_los_days

c = pd.read_parquet(INT / "cohort.parquet")
c = c[c.first_stay].copy()
# Restrict Table 1 to the ANALYTIC cohort (>=1 charted GCS-verbal entry) so its Ns reconcile
# exactly with Table 2 and the burden analysis. burden_per_stay holds the verbal-having first stays
# for BOTH neuro phenotypes and the comparator.
analytic = set(pd.read_parquet(INT / "burden_per_stay.parquet")["stay_id"])
c = c[c["stay_id"].isin(analytic)].copy()
vent = pd.read_parquet(INT / "vent.parquet")
c["vent"] = c["stay_id"].isin(vent.stay_id).astype(int)


def race_collapse(r):
    r = str(r).upper()
    if "WHITE" in r: return "White"
    if "BLACK" in r: return "Black"
    if "HISPANIC" in r or "LATINO" in r: return "Hispanic"
    if "ASIAN" in r: return "Asian"
    if r in ("UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER", "NAN"): return "Unknown/other"
    return "Unknown/other"


c["race_g"] = c["race"].map(race_collapse)
# Hospital length of stay, days: admission to the latest of the recorded discharge, death and ICU discharge times
# (in-hospital mortality is followed to hospital discharge).
c["hosp_los"] = hospital_los_days(c)
assert (c["hosp_los"].dropna() >= 0).all(), "negative hospital length of stay"
# Race is never a null value in MIMIC-IV; "not recorded" is an entry of unknown, unable to obtain or declined.
RACE_NOT_RECORDED = ("UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER")
c["race_missing"] = c["race"].isna() | c["race"].astype(str).str.upper().isin(RACE_NOT_RECORDED)
order = ["TBI", "ICH", "SAH", "SDH", "AIS", "anoxic", "comparator"]


def col(df):
    n = len(df)
    def pct(mask): return f"{mask.sum()} ({100*mask.mean():.1f})"
    out = {
        "N": n,
        "Age, median (IQR)": f"{df.age.median():.0f} ({df.age.quantile(.25):.0f}-{df.age.quantile(.75):.0f})",
        "Female, n (%)": pct(df.gender.eq("F")),
        "White, n (%)": pct(df.race_g.eq("White")),
        "Black, n (%)": pct(df.race_g.eq("Black")),
        "Medicare/Medicaid, n (%)": pct(df.insurance.isin(["Medicare", "Medicaid"])),
        "Mechanically ventilated, n (%)": pct(df.vent.eq(1)),
        "Hospital mortality, n (%)": pct(df.hospital_expire_flag.eq(1)),
        "ICU LOS days, median (IQR)": f"{df.los.median():.1f} ({df.los.quantile(.25):.1f}-{df.los.quantile(.75):.1f})",
        "Hospital LOS days, median (IQR)": f"{df.hosp_los.median():.1f} ({df.hosp_los.quantile(.25):.1f}-{df.hosp_los.quantile(.75):.1f})",
    }
    return pd.Series(out)


# Number of stays with a missing value, per Table 1 characteristic. Mechanical ventilation is derived from the
# ventilation-episode table, so a stay without an episode is "not ventilated" and cannot be missing.
MISSING = {
    "Age": lambda df: df.age.isna(),
    "Sex": lambda df: df.gender.isna(),
    "Race (null)": lambda df: df.race.isna(),
    "Race (unknown, unable to obtain or declined)": lambda df: df.race_missing,
    "Insurance": lambda df: df.insurance.isna(),
    "Mechanical ventilation": lambda df: df.vent.isna(),
    "Hospital mortality": lambda df: df.hospital_expire_flag.isna(),
    "ICU length of stay": lambda df: df.los.isna(),
    "Hospital length of stay": lambda df: df.hosp_los.isna(),
}


def missing(df):
    return pd.Series({k: int(f(df).sum()) for k, f in MISSING.items()})


tab = pd.DataFrame({ph: col(c[c.phenotype == ph]) for ph in order if (c.phenotype == ph).any()})
tab["Overall neuro"] = col(c[c.phenotype != "comparator"])
tab.to_csv(TAB / "table1.csv")
print(tab.to_string())

miss = pd.DataFrame({ph: missing(c[c.phenotype == ph]) for ph in order if (c.phenotype == ph).any()})
miss["Overall neuro"] = missing(c[c.phenotype != "comparator"])
miss.to_csv(TAB / "table1_missing.csv", index_label="Characteristic")
print(miss.to_string())
# Flat digest of the overall-neuro column so the counts quoted under Table 1 are checked like every other number.
overall = c[c.phenotype != "comparator"]
KEYS = {"Age": "age", "Sex": "sex", "Race (null)": "race_null",
        "Race (unknown, unable to obtain or declined)": "race_unrecorded", "Insurance": "insurance",
        "Mechanical ventilation": "ventilation", "Hospital mortality": "mortality",
        "ICU length of stay": "icu_los", "Hospital length of stay": "hospital_los"}
digest = {"n_overall": len(overall), "n_comparator": int((c.phenotype == "comparator").sum())}
for col_name, prefix in (("Overall neuro", "overall"), ("comparator", "comparator")):
    digest.update({f"{prefix}_{KEYS[k]}": int(v) for k, v in miss[col_name].items()})
(REV / "table1_missing.json").write_text(json.dumps(digest, indent=1, sort_keys=True), encoding="utf-8")
print(digest)
