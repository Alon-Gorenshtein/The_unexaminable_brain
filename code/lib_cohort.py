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
