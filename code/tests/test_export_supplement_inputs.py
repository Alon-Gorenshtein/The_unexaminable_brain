import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from config import IT, PROJ, REV


def _mod():
    spec = importlib.util.spec_from_file_location("exp", PROJ / "code" / "75_export_supplement_inputs.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_verbal_value_counts_on_synthetic_rows():
    chart = pd.DataFrame({"itemid": [IT["gcs_verbal"]] * 3 + [IT["gcs_eye"]],
                          "value": ["Oriented", "No Response-ETT", "Oriented", "None"]})
    got = _mod().verbal_value_counts(chart).set_index("value").n.to_dict()
    assert got == {"Oriented": 2, "No Response-ETT": 1}


def test_rass_missing_on_synthetic_rows():
    ex = pd.DataFrame({"rass": [np.nan, -2, np.nan, 0], "nonassess": [True, True, False, False],
                       "vent": [1, 1, 0, 0], "phenotype": ["TBI", "TBI", "AIS", "AIS"]})
    r = _mod().rass_missing(ex, ex.iloc[[0, 3]]).set_index("subgroup")
    assert r.loc["all", "rass_missing_n"] == 2 and r.loc["all", "n"] == 4
    assert r.loc["nonassess=True", "rass_missing_pct"] == 50
    assert r.loc["selected_all", "n"] == 2 and r.loc["selected_nonassess=True", "rass_missing_n"] == 1


@pytest.mark.requires_study_files("output/revision/rass_missing_by_subgroup.csv", "output/revision/aim4.json", "output/revision/aim4_rass_missing.csv")
def test_committed_rass_csv_agrees_with_the_secondary_model_outputs():
    r = pd.read_csv(REV / "rass_missing_by_subgroup.csv").set_index("subgroup")
    a4 = json.loads((REV / "aim4.json").read_text())
    old = pd.read_csv(REV / "aim4_rass_missing.csv", index_col=0)["rass_missing_pct"]
    for k, v in old.items():
        assert abs(r.loc[k, "rass_missing_pct"] - v) < 1e-9, k
    for k, key in [("all", "rass_missing_all_pct"), ("selected_all", "rass_missing_selected_pct"),
                   ("selected_nonassess=True", "rass_missing_selected_na_pct"),
                   ("selected_nonassess=False", "rass_missing_selected_assessable_pct")]:
        assert abs(r.loc[k, "rass_missing_pct"] - a4[key]) < 1e-9, k
    assert r.loc["all", "n"] == r.loc["nonassess=False", "n"] + r.loc["nonassess=True", "n"]
    assert r.loc["nonassess=False", "n"] == a4["n_exams"]
    assert (100 * r.rass_missing_n / r.n - r.rass_missing_pct).abs().max() < 1e-9


@pytest.mark.requires_study_files("output/revision/verbal_value_counts.csv")
def test_committed_verbal_counts_have_the_six_charted_values():
    v = pd.read_csv(REV / "verbal_value_counts.csv")
    assert set(v.value) == {"Oriented", "Confused", "Inappropriate Words", "Incomprehensible sounds",
                            "No Response", "No Response-ETT"}
    assert (v.n > 0).all() and v.n.is_monotonic_decreasing
