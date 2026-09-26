import pandas as pd, numpy as np, json
from config import INT, TAB

D = {}
burden = pd.read_parquet(INT / "burden_per_stay.parquet")
neuro = burden[burden.phenotype != "comparator"]
D["cohort"] = {
    "n_neuro_first_stay": int(len(neuro)),
    "by_phenotype": neuro.phenotype.value_counts().to_dict(),
    "n_comparator": int((burden.phenotype == "comparator").sum()),
}
D["aim1"] = {
    "overall_pct_ever_nonassess": round(100 * neuro.ever_nonassess.mean(), 1),
    "overall_median_frac_nonassess": round(100 * neuro.frac_nonassess.median(), 1),
    "table2": pd.read_csv(TAB / "table2_burden.csv").to_dict(orient="records"),
}
dark = pd.read_parquet(INT / "darktimes.parquet")
dn = dark[dark.phenotype != "comparator"]
D["aim2"] = {
    "median_t_first_dark_h": round(float(dn.t_first_dark.median()), 1),
    "concordance": pd.read_csv(TAB / "table_concordance.csv", index_col=0).to_dict(orient="index"),
}
res = pd.read_csv(TAB / "table3_aim3_by_phenotype.csv")
D["aim3"] = {
    "overall": res[res.phenotype == "ALL"].to_dict(orient="records"),
    "by_phenotype": res[res.phenotype != "ALL"].to_dict(orient="records"),
}
try:
    rc = pd.read_csv(TAB / "table_reclass_impute1_vs_kramer.csv", index_col=0)
    off = rc.values.sum() - np.trace(rc.values)
    D["aim3"]["reclass_pct_moved"] = round(100 * off / rc.values.sum(), 1)
except FileNotFoundError:
    pass

with open(INT.parent / "results_digest.json", "w") as f:
    json.dump(D, f, indent=2, default=str)
print(json.dumps(D, indent=2, default=str)[:2000])
print("\nWrote results_digest.json")
