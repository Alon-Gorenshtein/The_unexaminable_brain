import pandas as pd, numpy as np
from config import INT, TAB, ICU
from lib_assess import per_assessment_flags

cohort = pd.read_parquet(INT / "cohort.parquet")
cohort = cohort[cohort["first_stay"]]
icu = pd.read_csv(ICU / "icustays.csv.gz", usecols=["stay_id", "intime", "outtime"],
                  parse_dates=["intime", "outtime"])
chart = pd.read_parquet(INT / "chart_neuro.parquet")
flag = per_assessment_flags(chart)
flag = flag.merge(cohort[["stay_id", "phenotype"]], on="stay_id").merge(icu, on="stay_id")
flag["hr"] = (flag["charttime"] - flag["intime"]).dt.total_seconds() / 3600.0
flag = flag[(flag["hr"] >= 0) & (flag["hr"] <= 168)]
flag["hbin"] = flag["hr"].astype(int)

# ---- assessability curve: fraction assessable per (phenotype, hour) ----
g = (flag.groupby(["phenotype", "hbin"])
        .agg(n=("nonassess", "size"), na=("nonassess", "sum")).reset_index())
g["frac_assessable"] = 1 - g["na"] / g["n"]
g = g.rename(columns={"hbin": "hour", "n": "n_at_risk"})
g[["phenotype", "hour", "frac_assessable", "n_at_risk"]].to_parquet(
    INT / "assessability_curve.parquet", index=False)

# ---- per-stay dark times ----
def stay_times(df):
    na = df[df["nonassess"]]
    return pd.Series({"t_first_dark": na["hr"].min() if len(na) else np.nan,
                      "t_last_dark":  na["hr"].max() if len(na) else np.nan})
times = flag.groupby(["stay_id", "phenotype"]).apply(stay_times).reset_index()
times.to_parquet(INT / "darktimes.parquet", index=False)
print("Median time-to-first-dark (h) by phenotype:")
print(times.groupby("phenotype")["t_first_dark"].median().round(1).to_string())

# ---- concordance of non-assessable timestamps with sedative/NMB/vent ----
inf = pd.read_parquet(INT / "infusions.parquet")
vent = pd.read_parquet(INT / "vent.parquet")
na = flag[flag["nonassess"]][["stay_id", "charttime"]].copy()


def covered(na, iv):
    iv = iv[["stay_id", "starttime", "endtime"]]
    j = na.merge(iv, on="stay_id", how="left")
    hit = ((j["charttime"] >= j["starttime"] - pd.Timedelta("2h")) &
           (j["charttime"] <= j["endtime"] + pd.Timedelta("2h")))
    return j.assign(hit=hit).groupby(["stay_id", "charttime"])["hit"].any()


sed_hit = covered(na, inf[inf.cls == "sedative"]).rename("sedative")
nmb_hit = covered(na, inf[inf.cls == "nmb"]).rename("nmb")
vent_hit = covered(na, vent).rename("vent")
conc = pd.concat([sed_hit, nmb_hit, vent_hit], axis=1).reset_index().merge(
    cohort[["stay_id", "phenotype"]], on="stay_id")
ct = conc.groupby("phenotype")[["sedative", "nmb", "vent"]].mean().mul(100).round(1)
ct.to_csv(TAB / "table_concordance.csv")
print("\nConcordance of non-assessable exams with sedation/NMB/vent (%):")
print(ct.to_string())
