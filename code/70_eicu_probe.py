"""Stream nurseCharting once for 'GCS Total' rows, then count how many stays have a GCS charted
inside the first hours of documented mechanical ventilation. Also records the limits of the
ventilation anchor. Prints GO or NO-GO."""
import json
import os
import subprocess

from config import REV
from lib_eicu import (EICU, GCS_TSV, first_row_not_invasive, load_anchor, load_gcs, load_gcs_raw, load_treatment,
                      records_in_window)

GATE_MIN_STAYS = 5000   # pre-specified: GO if at least this many stays have a GCS Total in the first 6 h of ventilation

# nurseCharting columns (1-based): 2 patientunitstayid, 3 nursingchartoffset, 5 celltypecat ("Scores"),
# 6 celltypevallabel ("Glasgow coma score"), 7 celltypevalname ("GCS Total"), 8 nursingchartvalue
prog = '$6=="Glasgow coma score" && $7=="GCS Total" {print $2 "\\t" $3 "\\t" $8}'
if not GCS_TSV.exists():
    part = GCS_TSV.with_suffix(".tsv.part")
    with open(part, "w") as fh:
        gz = subprocess.Popen(["gzcat", f"{EICU}/nurseCharting.csv.gz"], stdout=subprocess.PIPE)
        aw = subprocess.Popen(["awk", "-F,", prog], stdin=gz.stdout, stdout=fh, env={**os.environ, "LC_ALL": "C"})
        gz.stdout.close()
        aw.communicate()
        gz.wait()
    if aw.returncode != 0 or gz.returncode != 0:
        raise SystemExit(f"stream failed: gzcat={gz.returncode} awk={aw.returncode}")
    part.rename(GCS_TSV)

raw = load_gcs_raw()
gcs = load_gcs()

tr = load_treatment()
print(tr[tr.treatmentstring.str.contains("mechanical ventilation")].treatmentstring.value_counts().head(8))
tv = load_anchor(tr)
w = records_in_window(gcs, tv, 0, 360)
w_stays = w.stay.unique()
out = dict(
    gate_min_stays=GATE_MIN_STAYS,
    n_stays_with_gcs=int(gcs.stay.nunique()),
    n_stays_with_vent=int(len(tv)),
    n_stays_gcs_in_first_6h_of_vent=int(len(w_stays)),
    # GCS Total rows as streamed: blank or non-numeric value, and numeric but outside 3-15
    n_gcs_rows_total=int(len(raw)),
    n_gcs_rows_nonnumeric=int(raw.total.isna().sum()),
    n_gcs_rows_numeric_outside_3_15=int((~raw.total.isna() & ~raw.total.between(3, 15)).sum()),
    # anchor limits. "Not invasive": the earliest ventilation row is tagged non-invasive and comes strictly before
    # every other ventilation row (or the stay has no other ventilation row); untagged stem rows count as invasive.
    n_stays_first_vent_row_not_invasive=int(len(first_row_not_invasive(tr))),
    n_stays_vent_offset_negative=int((tv < 0).sum()),
    n_stays_vent_offset_negative_gcs_window=int((tv.loc[w_stays] < 0).sum()),
    median_vent_offset_min_all_anchored=float(tv.median()),
    median_vent_offset_min_gcs_window=float(tv.loc[w_stays].median()),
)
out["decision"] = "GO" if out["n_stays_gcs_in_first_6h_of_vent"] >= GATE_MIN_STAYS else "NO-GO"
(REV / "eicu_probe.json").write_text(json.dumps(out, indent=1, allow_nan=False))
print(out)
