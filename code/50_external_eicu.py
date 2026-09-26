"""External replication on eICU-CRD v2.0 of the two transportable claims from the MIMIC-IV
non-assessable-verbal-GCS paper:

 (1) DEFAULT-TO-NORMAL CONVENTION. In MIMIC-IV the derived gcs.sql defaults intubated/non-verbal
     patients to a normal value. eICU stores the APACHE GCS components (apacheApsVar: eyes/motor/verbal,
     plus an `intubated` flag). A truly intubated patient cannot produce a verbal score of 5 ("oriented");
     any intubated stay carrying verbal=5 (GCS components summing to 15) is therefore a default-to-normal
     artifact, the direct analog of the MIMIC convention. We quantify how often intubated stays are
     assigned a normal verbal/total GCS.

 (2) SELECTION TRAP. Restricting to stays with an assessable/complete GCS (the implicit "complete-case"
     choice) drops the non-assessable stays. We test whether the dropped (non-assessable) stays are sicker
     (higher hospital mortality), which would bias any complete-case GCS analysis toward survivors.

Uses only apacheApsVar.csv.gz (GCS components + intubated/vent/meds) and patient.csv.gz (mortality,
admission dx). No large-table scan. eICU is multi-hospital, so this is a genuine out-of-database check."""
import gzip, csv, json, collections
from pathlib import Path
import numpy as np
from statsmodels.stats.proportion import proportion_confint

EICU = Path("/path/to/eicu-crd-2.0")
OUT = Path("./output")

def wil(k, n):
    if n == 0: return [None, None, None]
    lo, hi = proportion_confint(k, n, method="wilson")
    return [round(100*k/n, 1), round(100*lo, 1), round(100*hi, 1)]

# ---- apacheApsVar: one row per unit stay (APACHE worst-value window) ----
aps = {}
with gzip.open(EICU/"apacheApsVar.csv.gz", "rt") as f:
    for row in csv.DictReader(f):
        sid = row["patientunitstayid"]
        try:
            e, m, v = int(row["eyes"]), int(row["motor"]), int(row["verbal"])
            intub = int(row["intubated"]); vent = int(row["vent"]); meds = int(row["meds"])
        except (ValueError, KeyError):
            continue
        aps[sid] = dict(eyes=e, motor=m, verbal=v, intub=intub, vent=vent, meds=meds)

# ---- patient: mortality + neuro admission dx ----
NEURO = ("cva", "stroke", "hemorrhage", "haemorrhage", "subarachnoid", "intracerebral",
         "intracranial", "subdural", "head trauma", "head/brain", "neuro", "seizure",
         "encephalopathy", "coma", "cardiac arrest", "anoxic", "hypoxic", "hematoma")
pat = {}
with gzip.open(EICU/"patient.csv.gz", "rt") as f:
    for row in csv.DictReader(f):
        sid = row["patientunitstayid"]
        dx = (row.get("apacheadmissiondx") or "").lower()
        pat[sid] = dict(
            expired=(row.get("hospitaldischargestatus") == "Expired"),
            has_status=row.get("hospitaldischargestatus") in ("Alive", "Expired"),
            neuro=any(t in dx for t in NEURO),
        )

# ---- merge ----
rows = []
for sid, a in aps.items():
    p = pat.get(sid)
    if not p: continue
    assessable = a["verbal"] >= 1 and a["eyes"] >= 1 and a["motor"] >= 1  # -1 == not assessable
    total = (a["eyes"] + a["motor"] + a["verbal"]) if assessable else None
    rows.append(dict(sid=sid, **a, assessable=assessable, total=total,
                     expired=p["expired"], has_status=p["has_status"], neuro=p["neuro"]))

def analyze(rs, name):
    n = len(rs)
    intub = [r for r in rs if r["intub"] == 1]
    notintub = [r for r in rs if r["intub"] == 0]
    nonassess = [r for r in rs if not r["assessable"]]
    assess = [r for r in rs if r["assessable"]]
    # (1) default-to-normal: among intubated stays, fraction with verbal==5 and total GCS==15
    intub_v5 = sum(1 for r in intub if r["assessable"] and r["verbal"] == 5)
    intub_gcs15 = sum(1 for r in intub if r["assessable"] and r["total"] == 15)
    notintub_v5 = sum(1 for r in notintub if r["assessable"] and r["verbal"] == 5)
    # mortality with status known
    def mort(group):
        g = [r for r in group if r["has_status"]]
        d = sum(1 for r in g if r["expired"])
        return d, len(g), wil(d, len(g))
    d_non, n_non, m_non = mort(nonassess)
    d_ass, n_ass, m_ass = mort(assess)
    d_iv5, n_iv5, m_iv5 = mort([r for r in intub if r["assessable"] and r["verbal"] == 5])
    return {
        "name": name, "n_stays": n,
        "n_intubated": len(intub), "n_not_intubated": len(notintub),
        "n_nonassessable_gcs": len(nonassess),
        "pct_nonassessable_gcs": wil(len(nonassess), n),
        "default_to_normal": {
            "intubated_verbal5_pct": wil(intub_v5, len(intub)),
            "intubated_totalGCS15_pct": wil(intub_gcs15, len(intub)),
            "not_intubated_verbal5_pct": wil(notintub_v5, len(notintub)),
            "interpretation": "intubated patients cannot truly score verbal=5; this % is the default-to-normal artifact rate",
        },
        "selection_trap": {
            "mortality_nonassessable_pct": m_non, "n_nonassessable": n_non, "deaths_nonassessable": d_non,
            "mortality_assessable_pct": m_ass, "n_assessable": n_ass, "deaths_assessable": d_ass,
            "mortality_intubated_defaulted_normal_pct": m_iv5, "n_defaulted": n_iv5,
            "interpretation": "complete-case GCS analysis keeps 'assessable' and drops 'nonassessable'; higher mortality in dropped group = survivor-biased selection",
        },
    }

result = {
    "dataset": "eICU-CRD v2.0 (multi-hospital)",
    "n_stays_with_apache_and_patient": len(rows),
    "n_hospitals_note": "apacheApsVar covers the eICU APACHE-IVa subset across the multi-hospital cohort",
    "full_icu": analyze(rows, "Full ICU (all APACHE stays)"),
    "neuro_admission_subgroup": analyze([r for r in rows if r["neuro"]], "Neuro/arrest admission dx"),
    "mimic_reference": {
        "overall_pct_ever_nonassess_verbal": 45.2,
        "note": "MIMIC paper: gcs.sql defaults intubated to normal; complete-case (drop) selects survivors "
                "(drop strategy mean total GCS 13.32, lower mortality than impute strategies).",
    },
}
OUT.mkdir(parents=True, exist_ok=True)
json.dump(result, open(OUT/"external_eicu_digest.json", "w"), indent=2, default=float)
print(json.dumps(result, indent=2, default=float))
print("\nExternal eICU replication complete -> output/external_eicu_digest.json")
