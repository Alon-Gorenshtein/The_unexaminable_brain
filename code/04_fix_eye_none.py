"""The eye-opening value 'None' (score 1) was parsed as NaN when chartevents was staged, because pandas
treats the string 'None' as missing. valuenum is 1 on every such row, so the repair is exact.
The corrected table also takes RASS from at or before the examination (see attach_rass)."""
import pandas as pd

from config import INT
from lib_features import build_exam_table

c = pd.read_parquet(INT / "chart_neuro.parquet")
bad = (c.itemid == 220739) & c.value.isna()
assert int(bad.sum()) == 289687, int(bad.sum())
assert (c.loc[bad, "valuenum"] == 1).all()
assert c.loc[c.itemid.isin([223900, 223901]), "value"].notna().all()
c.loc[bad, "value"] = "None"
c.to_parquet(INT / "chart_neuro_fixed.parquet", index=False)
ex = build_exam_table(72, chart_file="chart_neuro_fixed.parquet", rass_direction="backward")
ex.to_parquet(INT / "aim4_exams_fixed.parquet", index=False)
print("eye None rows restored:", int(bad.sum()), "| exams:", len(ex), "| stays:", ex.stay_id.nunique())
