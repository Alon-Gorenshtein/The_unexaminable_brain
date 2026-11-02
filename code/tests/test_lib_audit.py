import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import lib_audit as A


def _exams():
    return pd.DataFrame({
        "stay_id": [1, 1, 2, 2],
        "charttime": pd.to_datetime(["2020-01-01 01:00", "2020-01-01 05:00",
                                     "2020-01-01 02:00", "2020-01-01 03:00"]),
        "eye": [1, 4, 4, 4], "motor": [2, 6, 6, 6],
        "verbal_score": [np.nan, 5.0, 5.0, np.nan],
        "nonassess": [True, False, False, True],
        "rass": [np.nan, -2.0, 0.0, np.nan],
        "y": [1, 1, 0, 0], "age": [60, 60, 40, 40],
    }).assign(em=lambda d: d.eye + d.motor)


def test_select_lowest_em_does_not_borrow_from_other_exams():
    r = A.select_exam(_exams(), "lowest_em").set_index("stay_id").loc[1]
    assert r.em == 3 and r.nonassess and np.isnan(r.rass) and np.isnan(r.verbal_score)
    naive = _exams().sort_values(["stay_id", "em", "charttime"]).groupby("stay_id").first()
    assert naive.loc[1, "rass"] == -2.0          # the bug this replaces is real


def test_rules_pick_the_expected_exam():
    d = _exams()
    assert A.select_exam(d, "earliest").set_index("stay_id").loc[2, "charttime"] == pd.Timestamp("2020-01-01 02:00")
    assert A.select_exam(d, "latest").set_index("stay_id").loc[2, "charttime"] == pd.Timestamp("2020-01-01 03:00")
    assert A.select_exam(d, "lowest_em").stay_id.tolist() == [1, 2]
    with pytest.raises(ValueError):
        A.select_exam(d, "bogus")


def test_totals_for_non_assessable_exams():
    w = A.add_totals(pd.DataFrame({"em": [2, 7, 10], "verbal_score": [np.nan] * 3, "nonassess": [True] * 3}))
    assert w.tg_derived.tolist() == [15, 15, 15]
    assert w.tg_impute1.tolist() == [3, 8, 11]
    assert w.tg_gcs_t.tolist() == [2, 7, 10]
    assert w.tg_brennan.tolist() == [3, 9, 15]
    assert w.tg_drop.isna().all()


def test_totals_agree_for_assessable_exam():
    w = A.add_totals(pd.DataFrame({"em": [8], "verbal_score": [4.0], "nonassess": [False]}))
    assert {float(w[c].iloc[0]) for c in ("tg_derived", "tg_impute1", "tg_gcs_t", "tg_brennan", "tg_drop")} == {12.0}


def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    a, b = A.boot_auc_ci(y, p), A.boot_auc_ci(y, p)
    assert a == b and a[0] < roc_auc_score(y, p) < a[1]


@pytest.mark.requires_study_files("output/intermediate/aim4_exams.parquet")
def test_real_data_legacy_population_reproduces_submitted_counts_and_missing_rass():
    w = A.select_exam(A.load_first_day(24, legacy=True), "lowest_em")
    assert len(w) == 12404 and w.stay_id.is_unique
    assert int(w.nonassess.sum()) == 3530
    assert w.loc[~w.nonassess, "verbal_score"].notna().all()
    assert int(w.rass.isna().sum()) == 3121      # 1,485 under the borrowing groupby().first()


@pytest.mark.requires_study_files("output/intermediate/aim4_exams_fixed.parquet")
def test_corrected_population_contains_eye_none_examinations_and_is_larger():
    d = A.load_first_day(24)
    assert (d.eye == 1).any()
    assert d.stay_id.nunique() >= 14000
    w = A.select_exam(d, "lowest_em")
    assert w.stay_id.is_unique and w.loc[~w.nonassess, "verbal_score"].notna().all()


def test_exposed_by_window():
    intime = pd.Series(pd.to_datetime(["2020-01-01 00:00"] * 4), index=[1, 2, 3, 4])
    ev = pd.DataFrame({"stay_id": [1, 2, 3, 9], "starttime": pd.to_datetime(
        ["2020-01-02 00:00", "2020-01-02 06:00", "2019-12-31 22:00", "2020-01-01 01:00"])})
    # exactly 24 h counts; 30 h does not; before admission counts; unknown stay ignored
    assert A.exposed_by(ev, intime) == {1, 3}


def test_pct_exposed_after_exam():
    intime = pd.Series(pd.to_datetime(["2020-01-01 00:00"] * 6), index=[1, 2, 3, 4, 5, 6])
    exam = pd.Series(pd.to_datetime(["2020-01-01 10:00"] * 6), index=[1, 2, 3, 4, 5, 6])
    ev = pd.DataFrame({"stay_id": [1, 1, 2, 3, 4, 5, 9], "starttime": pd.to_datetime(
        ["2020-01-01 12:00", "2020-01-01 08:00",      # stay 1: earliest start is before the examination
         "2020-01-01 11:00",                          # stay 2: starts after the examination
         "2020-01-01 10:00",                          # stay 3: starts at the examination, which is not after
         "2019-12-31 22:00",                          # stay 4: starts before admission, so before the examination
         "2020-01-02 06:00",                          # stay 5: 30 h after admission is not exposed in the first 24 h
         "2020-01-01 12:00"])})                       # stay 9: not in the audit
    # exposed in the first 24 h and in the audit: stays 1, 2, 3, 4; only stay 2 started after its examination
    assert A.pct_exposed_after_exam(ev, intime, exam) == pytest.approx(25.0)


def test_bounds_equal_brute_force_extremes():
    # Pool 2 extra deaths and 1 extra survivor into a 2-death, 3-survivor audit. In the worst case score a ranks every
    # new pair wrongly (new deaths lowest, new survivor highest) and score b rightly: the pooled difference is lo.
    # Swapping the two gives hi. All scores are distinct and finite, so every AUROC is exact.
    y_in = np.array([1, 1, 0, 0, 0])
    a_in = np.array([0.8, 0.6, 0.35, 0.4, 0.1])
    b_in = np.array([0.9, 0.5, 0.3, 0.7, 0.1])
    diff = roc_auc_score(y_in, a_in) - roc_auc_score(y_in, b_in)
    lo, hi, share = A.auroc_difference_bounds(diff, 2, 3, 2, 1)
    y, big = np.r_[y_in, 1, 1, 0], 1e6
    wrong, right = np.array([-big, -big + 1, big]), np.array([big, big - 1, -big])

    def pooled(a_new, b_new):
        return roc_auc_score(y, np.r_[a_in, a_new]) - roc_auc_score(y, np.r_[b_in, b_new])

    assert pooled(wrong, right) == pytest.approx(lo)
    assert pooled(right, wrong) == pytest.approx(hi)
    assert share == pytest.approx(10 / 16)          # 10 of the 16 pooled death-survivor pairs involve an excluded stay


def test_bounds_no_excluded_stays():
    assert A.auroc_difference_bounds(0.07, 100, 400, 0, 0) == pytest.approx((0.07, 0.07, 0.0))
