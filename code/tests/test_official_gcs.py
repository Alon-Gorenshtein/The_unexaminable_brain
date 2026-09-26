import json
import pandas as pd
import pytest

from config import ICU, REV

pytestmark = pytest.mark.requires_study_files(
    "output/revision/official_gcs.json",
    "output/revision/official_first_day_gcs.parquet",
    "output/revision/official_gcs_by_time.parquet")

COLS = {"stay_id", "gcs_min", "gcs_motor", "gcs_verbal", "gcs_eyes", "gcs_unable"}


def all_ett_windows_default_to_15(o, t, ic):
    j = t.merge(ic, on="stay_id")
    j = j[(j.charttime >= j.intime - pd.Timedelta("6h")) & (j.charttime <= j.intime + pd.Timedelta("24h"))]
    all_ett = j.groupby("stay_id")["gcs_unable"].min() == 1
    ids = all_ett[all_ett].index
    got = o.set_index("stay_id").loc[ids, "gcs_min"]
    return len(ids) > 0 and bool((got == 15).all())


def _load():
    o = pd.read_parquet(REV / "official_first_day_gcs.parquet")
    t = pd.read_parquet(REV / "official_gcs_by_time.parquet")
    ic = pd.read_csv(ICU / "icustays.csv.gz", usecols=["stay_id", "intime"], parse_dates=["intime"])
    return o, t, ic


def test_output_shape():
    o, _, _ = _load()
    assert COLS <= set(o.columns) and o.stay_id.is_unique
    assert o.gcs_min.dropna().between(3, 15).all()


def test_stays_with_only_non_assessable_rows_get_15_and_guard_trips_on_corruption():
    o, t, ic = _load()
    assert all_ett_windows_default_to_15(o, t, ic)
    j = t.merge(ic, on="stay_id")
    j = j[(j.charttime >= j.intime - pd.Timedelta("6h")) & (j.charttime <= j.intime + pd.Timedelta("24h"))]
    victim = (j.groupby("stay_id")["gcs_unable"].min() == 1).pipe(lambda s: s[s].index[0])
    broken = o.copy()
    broken.loc[broken.stay_id == victim, "gcs_min"] = 14
    assert not all_ett_windows_default_to_15(broken, t, ic)


def test_agreement_summary_written():
    j = json.loads((REV / "official_gcs.json").read_text())
    assert j["n_compared"] >= 14000 and j["legacy_n_compared"] > 12000
    assert j["agree_pct"] > 95                       # reconstruction from the corrected exam table
    assert j["legacy_agree_pct"] < j["agree_pct"]    # the legacy table lost the eye-opening rows
    assert j["official_higher_n"] == 0 and j["legacy_official_higher_n"] == 0
