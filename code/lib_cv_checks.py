"""Checks on the secondary model's cross-validation folds and RASS look-back, shared by script 65 and the tests."""


def patients_in_multiple_folds(df, group="subject_id", fold="fold"):
    """Number of patients whose rows fall in more than one fold; 0 when the folds are grouped by patient."""
    return int((df.groupby(group)[fold].nunique() > 1).sum())


def rass_lag_is_look_back(lag, window_h=2.0):
    """True when at least one RASS was matched and every matched value was charted from `window_h` hours before
    the examination up to the examination itself (`lag` = hours from the RASS to the examination; NaN = no match)."""
    lag = lag.dropna()
    return bool(len(lag) > 0 and lag.min() >= 0 and lag.max() <= window_h)
