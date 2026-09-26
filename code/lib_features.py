"""Shared feature engineering for the learned-imputation component (Aim 4)."""
import re
import numpy as np
import pandas as pd
from config import INT, ICU, IT, EYE_SCORE, MOTOR_SCORE, VERBAL_SCORE


def parse_rass(s):
    """RASS values are text beginning with a signed integer, e.g. '-2 Light sedation'."""
    m = re.search(r"[-+]?\d+", str(s))
    return int(m.group()) if m else np.nan


def attach_rass(trip, rass, direction="nearest", tolerance="2h"):
    """Attach one RASS value to each examination within its stay.

    direction="nearest": the value charted closest in time within the tolerance, on either side (the analysis
    as first submitted). direction="backward": the most recent value charted at or before the examination.
    Adds `rass` and `rass_lag_h`, the hours from the RASS to the examination (negative when the RASS was
    charted later); both are NaN when no value qualifies."""
    rass = rass.dropna(subset=["rass"]).sort_values("charttime")
    rass = rass[["stay_id", "charttime", "rass"]].assign(rass_time=lambda d: d["charttime"])
    out = pd.merge_asof(trip.sort_values("charttime"), rass, on="charttime", by="stay_id",
                        direction=direction, tolerance=pd.Timedelta(tolerance))
    out["rass_lag_h"] = (out["charttime"] - out["rass_time"]).dt.total_seconds() / 3600
    return out.drop(columns="rass_time")


def build_exam_table(hours=72, chart_file="chart_neuro.parquet", rass_direction="nearest"):
    """Per-(stay, charttime) GCS exams within `hours` of ICU admission for neuro first stays,
    with eye/motor scores, raw verbal, RASS within 2 h (nearest or look-back per `rass_direction`), context flags, and age.
    `chart_file` selects the staged chart table under output/intermediate.
    Returns one row per exam with columns:
      stay_id, charttime, hr, eye, motor, verbal_raw, verbal_score (NaN if non-assessable),
      nonassess, rass, rass_lag_h, sed, nmb, vaso, vent, age, phenotype, y(hospital_expire_flag)
    """
    cohort = pd.read_parquet(INT / "cohort.parquet")
    cohort = cohort[cohort.first_stay]
    cohort = cohort[cohort.phenotype != "comparator"]
    icu = pd.read_csv(ICU / "icustays.csv.gz", usecols=["stay_id", "intime"], parse_dates=["intime"])
    chart = pd.read_parquet(INT / chart_file)
    chart = chart[chart.stay_id.isin(cohort.stay_id)].merge(icu, on="stay_id")
    chart["hr"] = (chart["charttime"] - chart["intime"]).dt.total_seconds() / 3600
    chart = chart[(chart.hr >= 0) & (chart.hr <= hours)]

    eye = chart[chart.itemid == IT["gcs_eye"]].assign(eye=lambda x: x["value"].map(EYE_SCORE))
    mot = chart[chart.itemid == IT["gcs_motor"]].assign(motor=lambda x: x["value"].map(MOTOR_SCORE))
    vrb = chart[chart.itemid == IT["gcs_verbal"]].rename(columns={"value": "verbal_raw"})
    trip = (eye[["stay_id", "charttime", "hr", "eye"]]
            .merge(mot[["stay_id", "charttime", "motor"]], on=["stay_id", "charttime"])
            .merge(vrb[["stay_id", "charttime", "verbal_raw"]], on=["stay_id", "charttime"]))
    trip = trip.dropna(subset=["eye", "motor", "verbal_raw"])
    trip["verbal_score"] = trip["verbal_raw"].map(VERBAL_SCORE)
    trip["nonassess"] = trip["verbal_raw"].eq("No Response-ETT")

    rass = chart[chart.itemid == IT["rass"]][["stay_id", "charttime", "value"]].copy()
    rass["rass"] = rass["value"].map(parse_rass)
    trip = attach_rass(trip, rass, rass_direction)

    # context flags at exam time (active infusion/vent interval)
    inf = pd.read_parquet(INT / "infusions.parquet")
    vent = pd.read_parquet(INT / "vent.parquet")

    def active(df, iv):
        iv = iv[["stay_id", "starttime", "endtime"]]
        j = df[["stay_id", "charttime"]].merge(iv, on="stay_id", how="left")
        hit = (j["charttime"] >= j["starttime"]) & (j["charttime"] <= j["endtime"])
        return j.assign(hit=hit).groupby(["stay_id", "charttime"])["hit"].any()

    for name, sub in [("sed", inf[inf.cls == "sedative"]), ("nmb", inf[inf.cls == "nmb"]),
                      ("vaso", inf[inf.cls == "vasopressor"]), ("vent", vent)]:
        flag = active(trip, sub).rename(name)
        trip = trip.merge(flag, on=["stay_id", "charttime"], how="left")
        trip[name] = trip[name].fillna(False).astype(int)

    trip = trip.merge(cohort[["stay_id", "phenotype", "age", "hospital_expire_flag"]], on="stay_id")
    trip = trip.rename(columns={"hospital_expire_flag": "y"})
    return trip


# Production feature set for the Aim-4 verbal-recovery model. The mechanical-ventilation
# flag (`vent`) is deliberately EXCLUDED: it is near-deterministic for the non-assessable
# status (P[non-assessable | vent]=0.93) and is nearly constant in the assessable training
# set (1.4% vent-positive), so including it risks circularity when the model is applied to
# the intubated target. An ablation confirmed it carries no recovery signal (quadratic
# weighted kappa 0.562 with vent vs 0.561 without; downstream mortality AUROC 0.777 vs 0.781);
# it is dropped so the reported model cannot be accused of leaking the target. The with-vent
# variant is retained below for the reproducible ablation in 15a_aim4_ablation.py.
FEATURES = ["eye", "motor", "rass", "sed", "nmb", "vaso", "age"]
FEATURES_WITH_VENT = ["eye", "motor", "rass", "sed", "nmb", "vaso", "vent", "age"]
