import pandas as pd
from config import ICU, INT, SEDATIVES, NMB, VASOPRESSORS

LABEL = {**{k: ("sedative", v) for k, v in SEDATIVES.items()},
         **{k: ("nmb", v) for k, v in NMB.items()},
         **{k: ("vasopressor", v) for k, v in VASOPRESSORS.items()}}
want = set(LABEL)
cols = ["stay_id", "starttime", "endtime", "itemid", "amount", "rate"]
out = []
for ch in pd.read_csv(ICU / "inputevents.csv.gz", usecols=cols, chunksize=1_000_000,
                      parse_dates=["starttime", "endtime"]):
    ch = ch[ch["itemid"].isin(want)]
    if len(ch):
        out.append(ch)
inf = pd.concat(out, ignore_index=True)
inf["cls"] = inf["itemid"].map(lambda i: LABEL[i][0])
inf["label"] = inf["itemid"].map(lambda i: LABEL[i][1])
inf = inf[inf["endtime"] >= inf["starttime"]]
inf.to_parquet(INT / "infusions.parquet", index=False)
print("Wrote infusions:", len(inf), "rows")
print(inf.groupby("cls")["stay_id"].nunique().to_string())
