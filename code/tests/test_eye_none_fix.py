import pandas as pd
import pytest

from config import INT

pytestmark = pytest.mark.requires_study_files(
    "output/intermediate/chart_neuro.parquet",
    "output/intermediate/chart_neuro_fixed.parquet",
    "output/intermediate/aim4_exams.parquet",
    "output/intermediate/aim4_exams_fixed.parquet")


def test_fixed_chart_has_no_missing_eye_values():
    c = pd.read_parquet(INT / "chart_neuro_fixed.parquet", columns=["itemid", "value"])
    e = c[c.itemid == 220739]
    assert e.value.isna().sum() == 0 and int((e.value == "None").sum()) == 289687
    assert c[c.itemid.isin([223900, 223901])].value.notna().all()


def test_only_eye_rows_changed_and_valuenum_agrees():
    old = pd.read_parquet(INT / "chart_neuro.parquet")
    new = pd.read_parquet(INT / "chart_neuro_fixed.parquet")
    assert len(old) == len(new)
    changed = old.value.fillna("<NA>") != new.value.fillna("<NA>")
    assert set(new.loc[changed, "itemid"]) == {220739} and int(changed.sum()) == 289687
    assert (new.loc[changed, "valuenum"] == 1).all()


def test_fixed_exam_table_is_the_legacy_table_plus_eye_none_examinations():
    old = pd.read_parquet(INT / "aim4_exams.parquet")
    new = pd.read_parquet(INT / "aim4_exams_fixed.parquet")
    assert new.eye.notna().all() and set(new.eye.unique()) == {1, 2, 3, 4}
    assert (new.eye == 1).sum() > 0 and (old.eye == 1).sum() == 0
    assert (new.eye == 1).sum() <= 289687          # each such examination rests on a restored chart row
    key = ["stay_id", "charttime"]
    kept = new[new.eye >= 2].sort_values(key).reset_index(drop=True)
    # the legacy column is float64 (a NaN survived in the map before the rows were dropped); the
    # corrected column has no NaN and is int64. Only the storage type differs, so align it here and
    # keep every value and every other column's dtype under the exact comparison.
    kept["eye"] = kept["eye"].astype("float64")
    o = old.sort_values(key).reset_index(drop=True)
    # The corrected table takes RASS from at or before the examination (look-back), the legacy table from the
    # nearest value on either side, so `rass` and `rass_lag_h` legitimately differ; every other column must not.
    same = [c for c in o.columns if c != "rass"]
    pd.testing.assert_frame_equal(kept[same], o[same], check_exact=False, rtol=1e-9)
    assert set(kept.columns) - set(o.columns) == {"rass_lag_h"}
    # a look-back value implies a nearest value exists within the same window, so the corrected RASS can only
    # be missing more often, never present where the legacy one is missing
    assert not (kept.rass.notna() & o.rass.isna()).any()
    assert kept.rass.isna().sum() > o.rass.isna().sum()
