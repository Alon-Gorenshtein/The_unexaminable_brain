import pandas as pd, numpy as np
from config import INT, TAB

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
    }
    return pd.Series(out)


tab = pd.DataFrame({ph: col(c[c.phenotype == ph]) for ph in order if (c.phenotype == ph).any()})
tab["Overall neuro"] = col(c[c.phenotype != "comparator"])
tab.to_csv(TAB / "table1.csv")
print(tab.to_string())
