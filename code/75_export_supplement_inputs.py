"""Export two small committed inputs for the supplementary tables, so the table generator reads no git-ignored file.

output/revision/verbal_value_counts.csv: every charted value of the GCS verbal item across MIMIC-IV, with its count.
output/revision/rass_missing_by_subgroup.csv: examinations of the first 72 hours (and those selected for the audit)
with and without a RASS value within 2 hours, by subgroup."""
import pandas as pd

from config import INT, IT, REV
from lib_audit import load_exam_table, load_first_day, select_exam


def verbal_value_counts(chart):
    vc = chart.loc[chart["itemid"] == IT["gcs_verbal"], "value"].value_counts()
    return pd.DataFrame({"value": vc.index, "n": vc.values.astype(int)})


def _row(key, d):
    miss = int(d["rass"].isna().sum())
    return dict(subgroup=key, n=int(len(d)), rass_missing_n=miss, rass_missing_pct=100 * miss / len(d))


def rass_missing(ex, selected):
    na = ex["nonassess"].astype(bool)
    rows = [_row("all", ex), _row("nonassess=False", ex[~na]), _row("nonassess=True", ex[na]),
            _row("vent=0", ex[ex["vent"] == 0]), _row("vent=1", ex[ex["vent"] == 1])]
    rows += [_row(ph, g) for ph, g in ex.groupby("phenotype")]
    sna = selected["nonassess"].astype(bool)
    rows += [_row("selected_all", selected), _row("selected_nonassess=True", selected[sna]),
             _row("selected_nonassess=False", selected[~sna])]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    chart = pd.read_parquet(INT / "chart_neuro_fixed.parquet", columns=["itemid", "value"])
    verbal_value_counts(chart).to_csv(REV / "verbal_value_counts.csv", index=False)
    r = rass_missing(load_exam_table(), select_exam(load_first_day(24), "lowest_em"))
    r.to_csv(REV / "rass_missing_by_subgroup.csv", index=False)
    print(r.round(2).to_string(index=False))
