import numpy as np
import pandas as pd

from config import phenotype_masks

NEURO = ["SAH", "ICH", "SDH", "TBI", "AIS", "anoxic"]


def phenotype_flags(dx, hadm_ids):
    flags = pd.DataFrame(False, index=pd.Index(hadm_ids, name="hadm_id"), columns=NEURO)
    m = phenotype_masks(dx)
    for ph in NEURO:
        hit = flags.index.intersection(pd.Index(dx.loc[m[ph], "hadm_id"].unique()))
        flags.loc[hit, ph] = True
    return flags


def assign_hier(flags, low_to_high):
    lab = pd.Series(np.nan, index=flags.index, dtype=object)
    for ph in low_to_high:
        lab[flags[ph]] = ph
    return lab


def primary_dx_label(dx, hadm_ids):
    p = dx[dx["seq_num"] == 1]
    m = phenotype_masks(p)
    lab = pd.Series(np.nan, index=pd.Index(hadm_ids, name="hadm_id"), dtype=object)
    for ph in NEURO:
        hit = lab.index.intersection(pd.Index(p.loc[m[ph], "hadm_id"].unique()))
        lab[hit] = ph
    return lab


def hospital_los_days(df):
    """Days from hospital admission to the latest of the recorded hospital discharge, death and ICU discharge
    times. A few records (same-day deaths) carry a discharge time earlier than the admission time; taking the
    latest of the three end times keeps such a stay positive without dropping it."""
    ends = pd.concat([pd.to_datetime(df[c], errors="coerce") for c in ("dischtime", "deathtime", "outtime")],
                     axis=1).max(axis=1)
    return (ends - pd.to_datetime(df["admittime"], errors="coerce")).dt.total_seconds() / 86400
