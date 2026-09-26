"""Predictors of a non-assessable selected examination (logistic OR) and standardized mean differences,
complete-case excluded versus retained. Same specification as script 18, run on the corrected
population and, for reconciliation with the submitted tables, on the legacy population."""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import INT, REV
from lib_audit import add_stay_covariates, selected_frames

INF = pd.read_parquet(INT / "infusions.parquet", columns=["stay_id", "cls"])
SED_IDS = set(INF.loc[INF.cls == "sedative", "stay_id"])


def prepare(w):
    w = add_stay_covariates(w)
    w["vent"], w["vaso"] = w["vent_stay"], w["vaso_stay"]
    w["sed"] = w.stay_id.isin(SED_IDS).astype(int)
    w["na"] = w.nonassess.astype(int)
    w["died"] = w.y.astype(int)
    return w


def odds_ratios(w):
    dum = pd.get_dummies(w.phenotype, prefix="ph")
    cols = [c for c in dum.columns if c != "ph_AIS"]
    X = sm.add_constant(pd.concat([dum[cols].reset_index(drop=True),
                                  w[["age", "vent", "sed", "vaso", "died"]].reset_index(drop=True)],
                                 axis=1).astype(float))
    res = sm.Logit(w.na.values, X).fit(disp=0)
    ci = res.conf_int()
    rows = []
    for c in cols + ["age", "vent", "sed", "vaso", "died"]:
        rows.append(dict(predictor=c.replace("ph_", "") + (" vs AIS" if c.startswith("ph_") else ""),
                         OR=float(np.exp(res.params[c])), lo=float(np.exp(ci.loc[c, 0])),
                         hi=float(np.exp(ci.loc[c, 1])), p=float(res.pvalues[c])))
    return pd.DataFrame(rows)


def smd_table(w):
    exc, ret = w[w.na == 1], w[w.na == 0]

    def smd(col, binary):
        a, b = exc[col].astype(float), ret[col].astype(float)
        if binary:
            p1, p2 = a.mean(), b.mean()
            return (p1 - p2) / np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2 + 1e-9)
        return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-9)

    spec = [("Age (years)", "age", False), ("Mechanically ventilated", "vent", True),
            ("Sedative infusion", "sed", True), ("Vasopressor infusion", "vaso", True),
            ("Hospital mortality", "died", True)]
    rows = []
    for label, col, binary in spec:
        scale = 100 if binary else 1
        rows.append(dict(variable=label, excluded=scale * exc[col].astype(float).mean(),
                         retained=scale * ret[col].astype(float).mean(), smd=float(smd(col, binary))))
    return pd.DataFrame(rows)


frames = selected_frames(24)
out = {}
for label, key in [("corrected", "lowest_em"), ("legacy", "legacy_population_lowest_em")]:
    w = prepare(frames[key])
    orr, smd = odds_ratios(w), smd_table(w)
    orr.to_csv(REV / f"selection_model_{label}.csv", index=False)
    smd.to_csv(REV / f"smd_{label}.csv", index=False)
    for r in orr.itertuples():
        if r.predictor in ("vent", "sed", "vaso", "age", "died"):
            out.update({f"{label}_or_{r.predictor}": r.OR, f"{label}_or_{r.predictor}_lo": r.lo,
                        f"{label}_or_{r.predictor}_hi": r.hi})
    for r, k in zip(smd.itertuples(), ["age", "vent", "sed", "vaso", "died"]):
        out[f"{label}_smd_{k}"] = r.smd
        out[f"{label}_smd_{k}_excluded"] = r.excluded
        out[f"{label}_smd_{k}_retained"] = r.retained
    out[f"{label}_n_stays"] = int(len(w))
    out[f"{label}_n_excluded"] = int(w.na.sum())
(REV / "selection_model.json").write_text(json.dumps(out, indent=1, allow_nan=False))
print(pd.read_csv(REV / "selection_model_corrected.csv").round(2).to_string(index=False))
