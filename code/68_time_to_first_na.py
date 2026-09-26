"""Median time from ICU admission to the first non-assessable verbal examination, by phenotype, among stays that
became non-assessable. Reads output/intermediate/darktimes.parquet (built by 11_aim2_trajectory.py from verbal
entries only, so it does not depend on the eye-opening parsing) and writes a small table for the Results."""
import pandas as pd

from config import INT, REV

d = pd.read_parquet(INT / "darktimes.parquet").dropna(subset=["t_first_dark"])
out = (d.groupby("phenotype")["t_first_dark"].agg(n_stays="size", median_h="median")
       .reset_index().sort_values("phenotype"))
out.to_csv(REV / "time_to_first_non_assessable_by_phenotype.csv", index=False)
print(out.round(2).to_string(index=False))
