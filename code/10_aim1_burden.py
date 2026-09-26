import pandas as pd
from config import INT, TAB, ICU
from lib_assess import per_assessment_flags, per_stay_burden, burden_first_hours

cohort = pd.read_parquet(INT / "cohort.parquet")
icu = pd.read_csv(ICU / "icustays.csv.gz", usecols=["stay_id", "intime"], parse_dates=["intime"])
chart = pd.read_parquet(INT / "chart_neuro.parquet")

flag = per_assessment_flags(chart)
flag = flag[flag["stay_id"].isin(cohort["stay_id"])]
burden = per_stay_burden(flag)
for h in (24, 48, 72):
    burden = burden.merge(burden_first_hours(flag, icu, h), on="stay_id", how="left")

vent = pd.read_parquet(INT / "vent.parquet")
vent_ids = set(vent.stay_id)
m = cohort[["stay_id", "phenotype", "first_stay"]].merge(burden, on="stay_id", how="inner")
m = m[m["first_stay"]]                                  # primary analysis: first ICU stay
m["ventilated"] = m["stay_id"].isin(vent_ids)
m.to_parquet(INT / "burden_per_stay.parquet", index=False)


def summarize(df):
    vd = df[df["ventilated"]]
    return pd.Series({
        "n_stays": len(df),
        "pct_ventilated": round(100 * df["ventilated"].mean(), 1),
        "pct_ever_nonassess": round(100 * df["ever_nonassess"].mean(), 1),
        "median_frac_nonassess": round(100 * df["frac_nonassess"].median(), 1),
        "iqr_high": round(100 * df["frac_nonassess"].quantile(.75), 1),
        # ventilated subset (where the phenomenon lives):
        "vent_pct_ever_nonassess": round(100 * vd["ever_nonassess"].mean(), 1) if len(vd) else float("nan"),
        "vent_median_frac": round(100 * vd["frac_nonassess"].median(), 1) if len(vd) else float("nan"),
        "vent_median_frac_72h": round(100 * vd["frac_nonassess_72h"].median(), 1) if len(vd) else float("nan"),
        "median_n_verbal": df["n_verbal"].median(),
    })


order = ["TBI", "ICH", "SAH", "SDH", "AIS", "anoxic", "comparator"]
tab = m.groupby("phenotype").apply(summarize).reindex(order).reset_index()
tab.to_csv(TAB / "table2_burden.csv", index=False)
# overall neuro
neuro = m[m.phenotype != "comparator"]
print("OVERALL neuro first-stay N:", len(neuro),
      "| pct ever non-assessable: %.1f" % (100 * neuro["ever_nonassess"].mean()),
      "| median frac non-assessable: %.1f%%" % (100 * neuro["frac_nonassess"].median()))
print(tab.to_string(index=False))
