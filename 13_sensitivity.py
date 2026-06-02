"""Sensitivity analysis: broadened non-assessability definition =
'No Response-ETT' OR (verbal 'No Response' during an active mechanical-ventilation episode)."""
import pandas as pd
from config import INT, TAB, ICU, IT

cohort = pd.read_parquet(INT / "cohort.parquet")
cohort = cohort[cohort.first_stay]
chart = pd.read_parquet(INT / "chart_neuro.parquet")
vent = pd.read_parquet(INT / "vent.parquet")

v = chart[chart.itemid == IT["gcs_verbal"]][["stay_id", "charttime", "value"]].copy()
v = v[v.stay_id.isin(cohort.stay_id)]

# primary non-assessable
v["na_primary"] = v["value"].eq("No Response-ETT")
# broadened: also count "No Response" during an active vent episode
vt = vent[["stay_id", "starttime", "endtime"]]
j = v[v.value.eq("No Response")].merge(vt, on="stay_id", how="left")
inwin = (j["charttime"] >= j["starttime"]) & (j["charttime"] <= j["endtime"])
nr_vent = j.assign(inwin=inwin).groupby(["stay_id", "charttime"])["inwin"].any()
v = v.merge(nr_vent.rename("nr_during_vent"), on=["stay_id", "charttime"], how="left")
v["nr_during_vent"] = v["nr_during_vent"].fillna(False)
v["na_broad"] = v["na_primary"] | (v["value"].eq("No Response") & v["nr_during_vent"])

g = v.groupby("stay_id").agg(ever_primary=("na_primary", "any"),
                             ever_broad=("na_broad", "any")).reset_index()
g = g.merge(cohort[["stay_id", "phenotype"]], on="stay_id")
order = ["TBI", "ICH", "SAH", "SDH", "AIS", "anoxic", "comparator"]
tab = g.groupby("phenotype").agg(
    pct_ever_primary=("ever_primary", lambda s: round(100 * s.mean(), 1)),
    pct_ever_broad=("ever_broad", lambda s: round(100 * s.mean(), 1)),
).reindex(order)
tab.to_csv(TAB / "etable4_broadened_def.csv")
print(tab.to_string())
# confirm ordering of neuro phenotypes preserved
neuro = tab.drop("comparator")
print("\nOrdering preserved (primary vs broadened):",
      list(neuro.sort_values("pct_ever_primary", ascending=False).index) ==
      list(neuro.sort_values("pct_ever_broad", ascending=False).index))
