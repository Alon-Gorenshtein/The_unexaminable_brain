import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from config import REV
from lib_audit import calib_slope

STRATS = ("official", "reconstructed", "impute1", "gcs_t", "brennan")


def _reject(token):
    raise ValueError(f"bare {token} in comparability.json")


def _load():
    return json.loads((REV / "comparability.json").read_text(), parse_constant=_reject)


def test_slope_function_detects_overconfidence():
    rng = np.random.default_rng(0)
    true_p = 1 / (1 + np.exp(-rng.normal(-1, 1, 20000)))
    y = rng.binomial(1, true_p)
    over = 1 / (1 + np.exp(-2 * np.log(true_p / (1 - true_p))))
    assert abs(calib_slope(y, true_p) - 1) < 0.15 and calib_slope(y, over) < 0.7


@pytest.mark.requires_study_files("output/revision/comparability.json")
def test_groups_partition_the_common_set():
    j = _load()
    assert j["assessable_brennan_n"] + j["nonassessable_brennan_n"] == j["all_brennan_n"]


@pytest.mark.requires_study_files("output/revision/comparability.json")
def test_pooled_calibration_is_near_one_by_construction():
    j = _load()
    assert abs(j["all_brennan_oe_ratio"] - 1) < 0.02


@pytest.mark.requires_study_files("output/revision/comparability.json")
def test_json_has_no_bare_nan_or_infinity():
    j = _load()
    assert j and all(isinstance(v, (int, float)) for v in j.values())


@pytest.mark.requires_study_files("output/revision/comparability.json")
def test_strategies_share_the_same_patients_within_each_group():
    j = _load()
    for group in ("all", "nonassessable", "assessable"):
        for metric in ("n", "deaths"):
            vals = {s: j[f"{group}_{s}_{metric}"] for s in STRATS}
            assert len(set(vals.values())) == 1, (group, metric, vals)


@pytest.mark.requires_study_files("output/revision/comparability.json")
def test_complete_case_group_is_the_assessable_set():
    j = _load()
    for metric in ("n", "deaths"):
        ref = j[f"assessable_brennan_{metric}"]
        for s in ("complete_case", "brennan", "official"):
            assert j[f"cc_{s}_{metric}"] == ref, (metric, s)


@pytest.mark.requires_study_files("output/revision/comparability.json")
def test_official_calibration_direction_by_subgroup():
    j = _load()
    assert j["nonassessable_official_oe_ratio"] > 1
    assert j["assessable_official_oe_ratio"] < 1


@pytest.mark.requires_study_files("output/revision/comparability.json", "output/revision/audit_oof_lowest_em.parquet")
def test_nonassessable_brennan_auroc_recomputed_from_the_predictions():
    j = _load()
    oof = pd.read_parquet(REV / "audit_oof_lowest_em.parquet")
    d = oof[oof["nonassess"].astype(bool) & oof["p_brennan"].notna()]
    assert len(d) == j["nonassessable_brennan_n"]
    assert abs(roc_auc_score(d["y"].astype(int), d["p_brennan"]) - j["nonassessable_brennan_auroc"]) < 5e-7
