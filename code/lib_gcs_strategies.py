from config import VERBAL_SCORE, VERBAL_NONASSESS


def _kramer_add(em):
    """Kramer et al. J Neurosurg 2020 eye+motor -> verbal addition."""
    if em >= 10:
        return 5
    if em >= 8:
        return 4
    if em == 7:
        return 2
    return 1                 # em 2-6


def total_gcs(row, strategy):
    """Compute total GCS for one exam (dict eye, motor, verbal_raw) under a handling strategy.
    Returns None if the patient/observation is excluded by that strategy."""
    eye, motor, vraw = row["eye"], row["motor"], row["verbal_raw"]
    if eye is None or motor is None:
        return None
    nonassess = vraw in VERBAL_NONASSESS
    if not nonassess:
        v = VERBAL_SCORE.get(vraw)
        return None if v is None else eye + motor + v
    # non-assessable verbal:
    if strategy == "drop":
        return None
    if strategy == "impute1":
        return eye + motor + 1
    if strategy == "gcs_t":
        return eye + motor                       # E+M only (T convention)
    if strategy == "kramer":
        return eye + motor + _kramer_add(eye + motor)
    raise ValueError(strategy)
