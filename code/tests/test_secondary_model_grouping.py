import json

import numpy as np
import pandas as pd
import pytest

from config import INT, REV
from lib_cv_checks import patients_in_multiple_folds, rass_lag_is_look_back
from lib_features import attach_rass

STUDY = pytest.mark.requires_study_files(
    "output/intermediate/aim4_exams_fixed.parquet", "output/revision/aim4_oof.parquet", "output/revision/aim4.json")


def _exams(times):
    return pd.DataFrame({"stay_id": 1, "charttime": pd.to_datetime(times)})


def _rass(rows):
    return pd.DataFrame({"stay_id": 1, "charttime": pd.to_datetime([r[0] for r in rows]),
                         "rass": [r[1] for r in rows]})


def test_backward_never_takes_a_later_value():
    ex = _exams(["2020-01-01 10:00"])
    r = _rass([("2020-01-01 09:30", -3), ("2020-01-01 10:12", 0)])
    assert attach_rass(ex, r, "nearest").rass.iloc[0] == 0          # the later value is nearer
    got = attach_rass(ex, r, "backward")
    assert got.rass.iloc[0] == -3 and got.rass_lag_h.iloc[0] == pytest.approx(0.5)


def test_backward_is_missing_when_only_a_later_or_an_old_value_exists():
    ex = _exams(["2020-01-01 10:00"])
    assert np.isnan(attach_rass(ex, _rass([("2020-01-01 10:30", 1)]), "backward").rass.iloc[0])
    assert np.isnan(attach_rass(ex, _rass([("2020-01-01 07:59", -2)]), "backward").rass.iloc[0])


def test_a_value_charted_at_the_examination_time_counts():
    ex = _exams(["2020-01-01 10:00"])
    assert attach_rass(ex, _rass([("2020-01-01 10:00", -1)]), "backward").rass.iloc[0] == -1


def test_rass_is_matched_within_the_stay_only():
    ex = pd.concat([_exams(["2020-01-01 10:00"]), _exams(["2020-01-01 10:00"]).assign(stay_id=2)])
    r = _rass([("2020-01-01 09:50", -4)])
    got = attach_rass(ex, r, "backward").set_index("stay_id").rass
    assert got.loc[1] == -4 and np.isnan(got.loc[2])


def _folds(rows):
    return pd.DataFrame(rows, columns=["subject_id", "fold"])


def test_fold_check_counts_a_patient_present_in_two_folds():
    broken = _folds([(1, 0), (1, 1), (2, 1), (3, 2)])           # patient 1 is in folds 0 and 1
    assert patients_in_multiple_folds(broken) == 1
    assert patients_in_multiple_folds(_folds([(1, 0), (1, 1), (2, 0), (2, 3), (3, 2)])) == 2


def test_fold_check_passes_when_every_patient_has_one_fold():
    ok = _folds([(1, 0), (1, 0), (2, 1), (3, 2), (3, 2)])       # repeated rows of one patient stay in one fold
    assert patients_in_multiple_folds(ok) == 0


def test_fold_check_flags_folds_assigned_by_stay():
    # two stays of patient 7 fall in different folds: the failure of the stay-grouped scheme
    by_stay = pd.DataFrame({"subject_id": [7, 7, 8], "stay_id": [10, 11, 12], "fold": [0, 3, 1]})
    assert patients_in_multiple_folds(by_stay) == 1


def test_look_back_check_rejects_a_later_or_too_old_value():
    assert not rass_lag_is_look_back(pd.Series([0.0, 0.5, -0.2]))     # -0.2 h: the RASS was charted later
    assert not rass_lag_is_look_back(pd.Series([0.0, 1.0, 2.01]))     # older than the 2 h window
    assert not rass_lag_is_look_back(pd.Series([], dtype=float))      # nothing matched at all
    assert not rass_lag_is_look_back(pd.Series([np.nan, np.nan]))


def test_look_back_check_accepts_values_from_the_examination_back_to_two_hours():
    assert rass_lag_is_look_back(pd.Series([0.0, 0.5, 2.0, np.nan]))  # the window is closed at both ends


@STUDY
def test_corrected_exam_table_uses_only_earlier_rass():
    assert rass_lag_is_look_back(pd.read_parquet(INT / "aim4_exams_fixed.parquet").rass_lag_h)


@STUDY
def test_no_patient_is_in_two_folds_and_folds_are_balanced():
    oof = pd.read_parquet(REV / "aim4_oof.parquet")
    assert {"subject_id", "fold"} <= set(oof.columns)
    assert patients_in_multiple_folds(oof) == 0
    share = oof.fold.value_counts(normalize=True)
    assert share.min() > 0.15 and share.max() < 0.25


@STUDY
def test_the_cv_group_unit_is_recorded():
    a = json.loads((REV / "aim4.json").read_text())
    assert a["cv_group_unit"] == "subject_id" and a["n_cv_subjects"] > 0
