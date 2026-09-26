import json

import pandas as pd
import pytest

from config import REV, TAB

pytestmark = pytest.mark.requires_study_files(
    "output/revision/aim4.json",
    "output/revision/audit.json",
    "output/v1_submitted/table_aim4_recovery.csv")

V1 = TAB.parent / "v1_submitted"


def _j():
    return json.loads((REV / "aim4.json").read_text())


def test_legacy_examinations_reproduce_the_submitted_rule_based_recovery_numbers():
    v1 = pd.read_csv(V1 / "table_aim4_recovery.csv").set_index("method")
    j = _j()
    for key, old in [("brennan", "Brennan (eye+motor)"), ("fixed1", "Fixed verbal=1")]:
        assert abs(j[f"legacy_{key}_qwk"] - v1.loc[old, "qwk"]) < 1e-3
        assert abs(j[f"legacy_{key}_acc"] - v1.loc[old, "accuracy"]) < 1e-3


def test_patient_level_interval_is_wider_than_the_submitted_examination_level_one():
    j = _j()
    assert (j["learned_qwk_hi"] - j["learned_qwk_lo"]) > 3 * (j["learned_qwk_exam_hi"] - j["learned_qwk_exam_lo"])


def test_borrowed_rass_count_and_brennan_auroc_agrees_with_the_audit_table():
    j = _j()
    assert j["n_rass_borrowed_v1"] == 1636          # measured on the submitted population
    assert j["n_rass_borrowed_corrected_pop"] > 0
    a = json.loads((REV / "audit.json").read_text())
    assert abs(j["brennan_downstream_auroc"] - a["brennan_lowest_em_auroc"]) < 1e-12
    assert abs(j["brennan_downstream_lo"] - a["brennan_lowest_em_lo"]) < 1e-12
