"""(1) How many first-day patients received GCS 15 from the OFFICIAL derivation, and how many of
those had any non-assessable exam. (2) How often the default 15 is arithmetically impossible given
the charted eye and motor scores. Both need no imputation and no reference standard."""
import json

import numpy as np
import pandas as pd

from config import ICU, INT, REV
from lib_audit import load_first_day, select_exam, selected_frames
from lib_feasible import outside_total_bounds, sofa_bounds

o = pd.read_parquet(REV / "official_first_day_gcs.parquet")
t = pd.read_parquet(REV / "official_gcs_by_time.parquet")
ic = pd.read_csv(ICU / "icustays.csv.gz", usecols=["stay_id", "intime"], parse_dates=["intime"])
d1 = load_first_day(24)
coh = pd.read_parquet(INT / "cohort.parquet")
cohort_ids = set(coh.loc[coh.first_stay & (coh.phenotype != "comparator"), "stay_id"])
out = {}

# ---- official first_day_gcs, over the neuro first-stay cohort ----
o = o[o.stay_id.isin(cohort_ids) & o.gcs_min.notna()].copy()
win = t.merge(ic, on="stay_id")
win = win[(win.charttime >= win.intime - pd.Timedelta("6h")) & (win.charttime <= win.intime + pd.Timedelta("24h"))]
o["any_ett"] = o.stay_id.map(win.groupby("stay_id")["gcs_unable"].max().astype(bool)).fillna(False).astype(bool)
o15 = o[o.gcs_min == 15]
out["official_n"] = int(len(o))
out["official_15_n"] = int(len(o15))
out["official_15_pct"] = float(100 * len(o15) / len(o))
out["official_15_any_ett_n"] = int(o15.any_ett.sum())
out["official_15_no_ett_n"] = int((~o15.any_ett).sum())
out["any_ett_n"] = int(o.any_ett.sum())
out["any_ett_official_15_pct"] = float(100 * o[o.any_ett].gcs_min.eq(15).mean())

# the 15 was assigned by the default rule: a non-assessable entry in the window, including one carried
# forward from an earlier time (gcs_verbal == 0 is gcs.sql's code for No Response-ETT)
sel = o15[(o15.gcs_unable == 1) | (o15.gcs_verbal == 0)]
em = sel.gcs_eyes.fillna(4) + sel.gcs_motor.fillna(6)   # gcs.sql defaults for a missing component
imp = np.asarray(outside_total_bounds(em.values, 15))
out["official_default_15_n"] = int(len(sel))
out["official_default_15_impossible_n"] = int(imp.sum())
out["official_default_15_impossible_pct"] = float(100 * imp.mean())

# ---- examination-level rules from the exam table ----
for rule, w in selected_frames(24).items():
    na = w[w.nonassess]
    bad = np.asarray(outside_total_bounds(na.em.values, 15))
    least = np.array([sofa_bounds(e)[0] for e in na.em])
    out[f"{rule}_na_n"] = int(len(na))
    out[f"{rule}_na_impossible_n"] = int(bad.sum())
    out[f"{rule}_na_impossible_pct"] = float(100 * bad.mean())
    out[f"{rule}_na_sofa0_outside_n"] = int((least > 0).sum())
    out[f"{rule}_na_sofa_ge3_even_if_verbal5_n"] = int((least >= 3).sum())

w = select_exam(d1, "lowest_em")
dist = (w[w.nonassess].groupby("em").size().rename("n").reset_index()
        .assign(default_15_feasible=lambda x: x.em >= 10))
dist.to_csv(REV / "feasible_em_distribution.csv", index=False)
(REV / "feasible.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
