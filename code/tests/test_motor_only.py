import importlib.util
import json

import pandas as pd
import pytest

from config import PROJ, REV

pytestmark = pytest.mark.requires_study_files(
    "output/revision/motor_only.csv",
    "output/revision/motor_only.json",
    "output/revision/audit.json",
    "output/revision/audit_by_rule.csv",
    "output/intermediate/aim4_exams_fixed.parquet")


def _load():
    return pd.read_csv(REV / "motor_only.csv"), json.loads((REV / "motor_only.json").read_text())


def _module():
    spec = importlib.util.spec_from_file_location("motor_only", PROJ / "code" / "69_motor_only.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_brennan_reference_equals_the_canonical_audit_row():
    t, _ = _load()
    a = pd.read_csv(REV / "audit_by_rule.csv")
    canon = a[(a.rule == "lowest_em") & (a.slug == "brennan")].iloc[0]
    ref = t[(t.scope == "all") & (t.slug == "brennan")].iloc[0]
    for k in ("n", "deaths", "auroc", "lo", "hi", "brier"):
        assert abs(ref[k] - canon[k]) < 1e-12, k


def test_scopes_match_the_audit_subset_and_its_non_assessable_stays():
    t, s = _load()
    a = json.loads((REV / "audit.json").read_text())
    assert set(t[t.scope == "all"].n) == {a["sel_n_total"]}
    assert set(t[t.scope == "nonassessable"].n) == {a["sel_n_excluded"]}
    assert set(t[t.scope == "nonassessable"].deaths) == {a["sel_deaths_excluded"]}


def test_json_mirrors_the_table_and_intervals_bracket_estimates():
    t, s = _load()
    for r in t.itertuples():
        for k in ("n", "deaths", "auroc", "lo", "hi", "brier", "mean_value"):
            assert abs(s[f"{r.scope}_{r.slug}_{k}"] - float(getattr(r, k))) < 1e-12
        assert r.lo < r.auroc < r.hi
    for scope in ("all", "nonassessable"):
        assert s[f"{scope}_paired_brennan_minus_motor_lo"] < s[f"{scope}_paired_brennan_minus_motor_hi"]


def test_selection_recomputed_from_the_exam_table():
    t, s = _load()
    w = _module().selected()
    assert len(w) == t[(t.scope == "all") & (t.slug == "motor")].n.iloc[0]
    assert abs(w["motor_only"].mean() - t[(t.scope == "all") & (t.slug == "motor")].mean_value.iloc[0]) < 1e-9
    na = w[w["nonassess"].astype(bool)]
    assert abs(100 * na["nmb"].mean() - s["nonassessable_nmb_active_pct"]) < 1e-9
    assert int(na["nmb"].sum()) == s["nonassessable_nmb_active_n"]
    assert abs(100 * s["nonassessable_nmb_active_n"] / len(na) - s["nonassessable_nmb_active_pct"]) < 1e-9
    assert w["motor_only"].between(1, 6).all()
