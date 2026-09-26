import pandas as pd
from config import IT, VERBAL_NONASSESS


def per_assessment_flags(chart_verbal):
    """Input: chartevents rows. Returns GCS-verbal rows with a 'nonassess' bool."""
    v = chart_verbal[chart_verbal["itemid"] == IT["gcs_verbal"]].copy()
    v["nonassess"] = v["value"].isin(VERBAL_NONASSESS)
    return v.sort_values(["stay_id", "charttime"])


def per_stay_burden(flagged):
    """Per-stay verbal-assessment burden metrics."""
    g = flagged.groupby("stay_id")
    out = g.agg(n_verbal=("nonassess", "size"),
                n_nonassess=("nonassess", "sum"),
                first_time=("charttime", "min"),
                last_time=("charttime", "max")).reset_index()
    out["frac_nonassess"] = out["n_nonassess"] / out["n_verbal"]
    out["ever_nonassess"] = out["n_nonassess"] > 0
    return out


def burden_first_hours(flagged, cohort_intime, hours):
    """Fraction non-assessable among verbal assessments within first `hours` of ICU intime."""
    f = flagged.merge(cohort_intime, on="stay_id", how="inner")
    f["hr"] = (f["charttime"] - f["intime"]).dt.total_seconds() / 3600.0
    w = f[(f["hr"] >= 0) & (f["hr"] <= hours)]
    g = w.groupby("stay_id")["nonassess"].agg(["size", "sum"]).reset_index()
    g[f"frac_nonassess_{hours}h"] = g["sum"] / g["size"]
    return g[["stay_id", f"frac_nonassess_{hours}h"]]
