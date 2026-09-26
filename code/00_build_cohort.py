import pandas as pd
from config import HOSP, ICU, INT, phenotype_masks, PHENOTYPE_PRIORITY

dx = pd.read_csv(HOSP / "diagnoses_icd.csv.gz", dtype={"icd_code": str, "icd_version": "Int64"})
dx["icd_code"] = dx["icd_code"].str.strip().str.upper()
masks = phenotype_masks(dx)

# Map each hadm_id to a single phenotype. Iterate low->high priority so the
# highest-priority phenotype (last in PHENOTYPE_PRIORITY = SAH) wins.
hadm_pheno = {}
for name in PHENOTYPE_PRIORITY:
    for h in dx.loc[masks[name], "hadm_id"].unique():
        hadm_pheno[h] = name

icu = pd.read_csv(ICU / "icustays.csv.gz", parse_dates=["intime", "outtime"])
icu = icu.sort_values(["subject_id", "hadm_id", "intime"])
icu["first_stay"] = ~icu.duplicated(["hadm_id"])           # first ICU stay per hospitalization
icu["phenotype"] = icu["hadm_id"].map(hadm_pheno).fillna("comparator")

pat = pd.read_csv(HOSP / "patients.csv.gz")[["subject_id", "gender", "anchor_age", "dod"]]
adm = pd.read_csv(HOSP / "admissions.csv.gz")[
    ["hadm_id", "race", "insurance", "language", "marital_status",
     "hospital_expire_flag", "deathtime", "dischtime", "admittime", "discharge_location"]]

c = (icu.merge(pat, on="subject_id", how="left")
        .merge(adm, on="hadm_id", how="left")
        .rename(columns={"anchor_age": "age"}))
c = c[c["age"] >= 18].copy()
c["first_stay"] = c["first_stay"].astype(bool)
c.to_parquet(INT / "cohort.parquet", index=False)

n_neuro = (c["phenotype"] != "comparator").sum()
print("Wrote cohort:", len(c), "stays;", n_neuro, "neuro")
print(c["phenotype"].value_counts().to_string())
print("\nFirst-stay neuro by phenotype:")
print(c[c.first_stay & (c.phenotype != "comparator")]["phenotype"].value_counts().to_string())
