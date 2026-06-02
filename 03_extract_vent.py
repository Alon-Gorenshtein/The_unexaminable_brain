import pandas as pd
from config import ICU, INT

pe = pd.read_csv(ICU / "procedureevents.csv.gz",
                 usecols=["stay_id", "itemid", "starttime", "endtime"],
                 parse_dates=["starttime", "endtime"])
vent = pe[pe["itemid"] == 225792][["stay_id", "starttime", "endtime"]].copy()  # Invasive Ventilation
vent = vent[vent["endtime"] >= vent["starttime"]]
vent.to_parquet(INT / "vent.parquet", index=False)
print("Wrote vent episodes:", len(vent), "covering", vent.stay_id.nunique(), "stays")
