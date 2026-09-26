import subprocess, csv, os, pandas as pd
from config import ICU, INT, CHART_ITEMIDS, IT

ids = ",".join(str(i) for i in sorted(CHART_ITEMIDS))
tsv = INT / "chart_neuro_raw.tsv"
# chartevents cols: 1 subject,2 hadm,3 stay,4 caregiver,5 charttime,6 storetime,
#                   7 itemid,8 value,9 valuenum,10 valueuom,11 warning
# Emit TAB-separated (values are tab-free) -> avoids CSV quote/comma traps entirely.
awk = (r'BEGIN{split("%s",a,",");for(i in a)w[a[i]]=1} '
       r'NR==1{next} ($7 in w){print $1"\t"$3"\t"$5"\t"$7"\t"$8"\t"$9}') % ids
cmd = f'gzcat "{ICU/"chartevents.csv.gz"}" | LC_ALL=C awk -F, \'{awk}\' > "{tsv}"'

print("Streaming chartevents (one full pass, ~10-15 min)...", flush=True)
r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
if r.returncode != 0:
    raise RuntimeError("pipeline failed: " + r.stderr[:500])

ce = pd.read_csv(tsv, sep="\t", header=None, quoting=csv.QUOTE_NONE, dtype=str,
                 names=["subject_id", "stay_id", "charttime", "itemid", "value", "valuenum"],
                 on_bad_lines="skip")
ce["itemid"] = pd.to_numeric(ce["itemid"], errors="coerce")
ce["stay_id"] = pd.to_numeric(ce["stay_id"], errors="coerce")
ce["valuenum"] = pd.to_numeric(ce["valuenum"], errors="coerce")
ce["charttime"] = pd.to_datetime(ce["charttime"], errors="coerce")
ce = ce.dropna(subset=["stay_id", "itemid"]).astype({"stay_id": "int64", "itemid": "int64"})
ce.to_parquet(INT / "chart_neuro.parquet", index=False)
os.remove(tsv)
print("Wrote chart_neuro:", len(ce), "rows", flush=True)

v = ce[ce.itemid == IT["gcs_verbal"]]
print("\nGCS-verbal value counts (full data):")
print(v["value"].value_counts(dropna=False).to_string())
print("\nverbal-ETT fraction = %.3f" % (v["value"] == "No Response-ETT").mean())
