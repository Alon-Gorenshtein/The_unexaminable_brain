from pathlib import Path

DATA = Path("/path/to/mimic-iv-3.1")
HOSP, ICU = DATA / "hosp", DATA / "icu"
PROJ = Path(".")
INT = PROJ / "output" / "intermediate"
TAB = PROJ / "output" / "tables"
FIG = PROJ / "output" / "figures"
for d in (INT, TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)

# --- chartevents itemids ---
IT = dict(
    gcs_eye=220739, gcs_verbal=223900, gcs_motor=223901,
    rass=228096, code_status=223758,
    pupil_size_r=223907, pupil_size_l=224733, pupil_resp_r=227121, pupil_resp_l=227288,
    temp_f=223761, temp_c=223762, vent_mode=223849, mech_vent=226260,
)
CHART_ITEMIDS = set(IT.values())

# Non-assessability value set for GCS-verbal (223900). Primary token confirmed in data.
VERBAL_NONASSESS = {"No Response-ETT"}              # primary definition (intubation/trach)
VERBAL_SCORE = {                                    # assessable verbal values
    "Oriented": 5, "Confused": 4, "Inappropriate Words": 3,
    "Incomprehensible sounds": 2, "No Response": 1,
}
EYE_SCORE = {
    "Spontaneously": 4, "To Speech": 3, "To Pain": 2, "None": 1,
}
MOTOR_SCORE = {
    "Obeys Commands": 6, "Localizes Pain": 5, "Flex-withdraws": 4,
    "Abnormal Flexion": 3, "Abnormal extension": 2, "No response": 1, "No Response": 1,
}

# --- inputevents itemids ---
SEDATIVES = {222168: "propofol", 221668: "midazolam", 225150: "dexmedetomidine",
             229420: "dexmedetomidine", 221744: "fentanyl", 225942: "fentanyl",
             225972: "fentanyl_push", 221712: "ketamine", 221385: "lorazepam",
             225156: "pentobarbital"}
NMB = {221555: "cisatracurium", 222062: "vecuronium", 229233: "rocuronium"}
VASOPRESSORS = {221906: "norepinephrine", 222315: "vasopressin", 221749: "phenylephrine",
                221662: "dopamine", 221289: "epinephrine"}
INPUT_ITEMIDS = set(SEDATIVES) | set(NMB) | set(VASOPRESSORS)

# --- procedureevents itemids ---
VENT_PROC = {225792: "invasive_vent", 224385: "intubation", 227194: "extubation"}

# --- ICD phenotype definitions (icd_version-aware) ---
# Listed low -> high priority; last write wins => SAH highest priority.
PHENOTYPE_PRIORITY = ["anoxic", "AIS", "TBI", "SDH", "ICH", "SAH"]


def phenotype_masks(df):
    """df has columns icd_code (UPPER, stripped) and icd_version (int).
    Returns dict name -> boolean Series."""
    c, v = df["icd_code"], df["icd_version"]
    is9, is10 = v == 9, v == 10
    return {
        "SAH":    (is9 & c.str.startswith("430")) | (is10 & c.str.startswith("I60")),
        "ICH":    (is9 & c.str.startswith("431")) | (is10 & c.str.startswith("I61")),
        "SDH":    (is9 & c.str.startswith("432")) | (is10 & c.str.startswith("I62")),
        "AIS":    (is9 & c.str.match(r'^43[34]')) | (is10 & c.str.startswith("I63")),
        "TBI":    (is9 & c.str.match(r'^(80[0-4]|85[0-4])')) | (is10 & c.str.startswith("S06")),
        "anoxic": (is9 & c.str.startswith("3481")) | (is10 & c.str.startswith("G931"))
                  | (is9 & c.str.startswith("3485")) | (is10 & c.str.startswith("G9382")),
    }
