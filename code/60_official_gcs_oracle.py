"""Run the official mimic-code gcs.sql and first_day_gcs.sql (DuckDB translation, pinned commit in
code/vendor/mimic_code/PIN.txt) on the staged chartevents. Run with a separate DuckDB environment built from
code/requirements-duckdb.txt, for example:
    ~/.venvs/unexam_duckdb/bin/python code/60_official_gcs_oracle.py"""
import json

import duckdb
import numpy as np
import pandas as pd

from config import ICU, INT, REV, VENDOR

con = duckdb.connect()
con.execute("CREATE SCHEMA mimiciv_icu; CREATE SCHEMA mimiciv_derived;")
con.execute(f"""CREATE TABLE mimiciv_icu.chartevents AS
    SELECT subject_id, stay_id, charttime, itemid, value, valuenum
    FROM read_parquet('{INT / 'chart_neuro.parquet'}')""")
con.execute(f"CREATE TABLE mimiciv_icu.icustays AS SELECT * FROM read_csv_auto('{ICU / 'icustays.csv.gz'}')")
for f in ("gcs.sql", "first_day_gcs.sql"):
    con.execute((VENDOR / f).read_text())

first_day = con.execute("""SELECT stay_id, gcs_min, gcs_motor, gcs_verbal, gcs_eyes, gcs_unable
                           FROM mimiciv_derived.first_day_gcs""").df()
by_time = con.execute("""SELECT stay_id, charttime, gcs, gcs_motor, gcs_verbal, gcs_eyes, gcs_unable
                         FROM mimiciv_derived.gcs""").df()
first_day.to_parquet(REV / "official_first_day_gcs.parquet", index=False)
by_time.to_parquet(REV / "official_gcs_by_time.parquet", index=False)



def agreement(exam_file):
    """Agreement of the official first-day minimum with the reconstruction (first-day minimum of the
    simple rule) built from one exam table."""
    ex = pd.read_parquet(INT / exam_file)
    d = ex[ex.hr <= 24].dropna(subset=["eye", "motor"]).copy()
    d["em"] = d.eye + d.motor
    d["derived"] = np.where(d.nonassess, 15, d.em + d.verbal_score)
    recon = d.groupby("stay_id")["derived"].min().rename("recon")
    cmp = first_day.set_index("stay_id").join(recon, how="inner").dropna(subset=["gcs_min", "recon"])
    diff = cmp.gcs_min - cmp.recon
    return dict(n_compared=int(len(cmp)), agree_pct=float(100 * (diff == 0).mean()),
                official_lower_n=int((diff < 0).sum()), official_higher_n=int((diff > 0).sum()),
                mean_abs_diff=float(diff.abs().mean()))


# corrected exam table (examinations with no eye opening kept) under the plain key names; the legacy
# table (the submitted reconstruction, which lost those examinations) under legacy_* keys
out = agreement("aim4_exams_fixed.parquet")
out.update({f"legacy_{k}": v for k, v in agreement("aim4_exams.parquet").items()})
(REV / "official_gcs.json").write_text(json.dumps(out, indent=1))
print(out)
