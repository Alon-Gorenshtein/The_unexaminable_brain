import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from config import EICU_DATA, INT

EICU = str(EICU_DATA)
GCS_TSV = INT / "eicu_gcs_nurse.tsv"

VENT_STEM = "pulmonary|ventilation and oxygenation|mechanical ventilation"
# Rows under the stem that are tagged non-invasive. Every other row under the stem (the bare stem and its
# invasive sub-strings) is counted as invasive, because the stem itself is the generic mechanical-ventilation label.
NONINVASIVE_TAG = "non-invasive"
# copied verbatim from 50_external_eicu.py so the neuro subgroup is the same one
NEURO_DX_TERMS = ("cva", "stroke", "hemorrhage", "haemorrhage", "subarachnoid", "intracerebral",
                  "intracranial", "subdural", "head trauma", "head/brain", "neuro", "seizure",
                  "encephalopathy", "coma", "cardiac arrest", "anoxic", "hypoxic", "hematoma")

# name: (lower bound, upper bound, both inclusive, in minutes from the first documented ventilation;
#        which record per stay). "first" keeps the earliest record in the window, "last" the latest.
# Post windows: the earliest record is the one nearest the anchor. Pre window: the latest record is the one
# nearest the anchor; pre_6h_earliest is the record furthest from it, kept to show how much that choice matters.
WINDOWS = {
    "w0_6h": (0, 360, "first"),
    "w1_6h": (60, 360, "first"),
    "w0_12h": (0, 720, "first"),
    "pre_6h": (-360, -1, "last"),
    "pre_6h_earliest": (-360, -1, "first"),
}


def first_ventilation_offset(treatment, invasive_only=False):
    t = treatment[treatment["treatmentstring"].str.startswith(VENT_STEM)]
    if invasive_only:
        t = t[~t["treatmentstring"].str.contains(NONINVASIVE_TAG, regex=False)]
    return t.groupby("patientunitstayid")["treatmentoffset"].min()


def first_row_not_invasive(treatment):
    """Stays whose earliest ventilation row is strictly earlier than every invasive row, or that have no
    invasive row at all (the earliest documented ventilation is a non-invasive-tagged row)."""
    tv = first_ventilation_offset(treatment)
    tinv = first_ventilation_offset(treatment, invasive_only=True).reindex(tv.index)
    return tv.index[(tv < tinv) | tinv.isna()]


def records_in_window(gcs, tv, lo_min, hi_min):
    m = gcs.merge(tv.rename("tv"), left_on="stay", right_index=True)
    m["rel"] = m["offset"] - m["tv"]
    return m[(m["rel"] >= lo_min) & (m["rel"] <= hi_min)]


def first_record_per_stay(rec):
    return rec.sort_values(["stay", "offset"]).drop_duplicates("stay")


def last_record_per_stay(rec):
    return rec.sort_values(["stay", "offset"]).drop_duplicates("stay", keep="last")


def window_records(gcs, tv, name):
    """One record per stay for the named window in WINDOWS (nearest to the anchor unless the name says earliest)."""
    lo, hi, how = WINDOWS[name]
    rec = records_in_window(gcs, tv, lo, hi)
    return last_record_per_stay(rec) if how == "last" else first_record_per_stay(rec)


def wilson_pct(k, n):
    if n == 0:
        return (float("nan"),) * 3
    lo, hi = proportion_confint(k, n, method="wilson")
    return 100 * k / n, 100 * lo, 100 * hi


def load_gcs_raw(path=GCS_TSV):
    """Every 'GCS Total' row streamed from nurseCharting; total is numeric or NaN (blank or non-numeric value)."""
    g = pd.read_csv(path, sep="\t", header=None, names=["stay", "offset", "total"], dtype=str)
    g["total"] = pd.to_numeric(g["total"], errors="coerce")
    return g


def load_gcs(path=GCS_TSV):
    """GCS Total rows with a numeric value from 3 to 15; stay and offset as ints."""
    g = load_gcs_raw(path).dropna().astype({"stay": int, "offset": int})
    return g[g.total.between(3, 15)]


def load_treatment():
    return pd.read_csv(f"{EICU}/treatment.csv.gz", usecols=["patientunitstayid", "treatmentoffset", "treatmentstring"])


def load_anchor(treatment=None):
    """First documented mechanical-ventilation offset per stay, indexed by patientunitstayid."""
    return first_ventilation_offset(load_treatment() if treatment is None else treatment)
