import json

import numpy as np
import pandas as pd
import pytest

from config import HOSP, INT, PHENOTYPE_PRIORITY, REV, TAB
from lib_cohort import NEURO, assign_hier, phenotype_flags


def _flags():
    f = pd.DataFrame(False, index=pd.Index([10, 11, 12], name="hadm_id"), columns=NEURO)
    f.loc[10, ["SAH", "AIS"]] = True
    f.loc[11, "AIS"] = True
    return f


def test_hierarchy_last_wins_and_reversal_only_moves_overlaps():
    a, b = assign_hier(_flags(), PHENOTYPE_PRIORITY), assign_hier(_flags(), PHENOTYPE_PRIORITY[::-1])
    assert a.loc[10] == "SAH" and b.loc[10] == "AIS"
    assert a.loc[11] == b.loc[11] == "AIS" and pd.isna(a.loc[12])


@pytest.mark.requires_study_files("output/intermediate/cohort.parquet")
def test_primary_rule_reproduces_submitted_phenotypes():
    c = pd.read_parquet(INT / "cohort.parquet")
    neuro = c[c.first_stay & (c.phenotype != "comparator")]
    dx = pd.read_csv(HOSP / "diagnoses_icd.csv.gz", usecols=["hadm_id", "icd_code", "icd_version"], dtype={"icd_code": str})
    dx["icd_code"] = dx.icd_code.str.strip().str.upper()
    dx = dx[dx.hadm_id.isin(neuro.hadm_id)]
    lab = assign_hier(phenotype_flags(dx, neuro.hadm_id.unique()), PHENOTYPE_PRIORITY)
    got = neuro.set_index("hadm_id").phenotype
    assert (lab.reindex(got.index) == got).all()


@pytest.mark.requires_study_files("output/revision/cohort_accounting.json", "output/revision/hierarchy_sensitivity.csv", "output/v1_submitted/table2_burden.csv")
def test_accounting_adds_up_and_matches_submitted_burden():
    j = json.loads((REV / "cohort_accounting.json").read_text())
    assert j["neuro_first_stay_n"] == j["with_verbal_n"] + j["no_verbal_n"]
    assert j["with_verbal_n"] == 14230
    assert j["with_verbal_n"] == j["aim3_n"] + j["aim3_excluded_n"]
    assert j["aim3_legacy_n"] == 12404 < j["aim3_n"]      # the corrected audit subset is larger than the submitted one
    assert j["aim3_legacy_excluded_n"] == j["with_verbal_n"] - j["aim3_legacy_n"]
    assert j["aim3_excl_first_verbal_after24h_n"] + j["aim3_excl_verbal_le24h_no_triple_n"] == j["aim3_excluded_n"]
    assert j["overlap_1_n"] + j["overlap_2_n"] + j["overlap_3plus_n"] == j["overlap_total_n"]
    h = pd.read_csv(REV / "hierarchy_sensitivity.csv")
    prim = h[h.rule == "primary"].set_index("phenotype").ever_na_pct
    t2 = pd.read_csv(TAB.parent / "v1_submitted" / "table2_burden.csv").set_index("phenotype").pct_ever_nonassess
    for ph in NEURO:
        assert abs(prim[ph] - t2[ph]) < 0.06


OUTSIDE_FILES = ("output/revision/cohort_accounting.json", "output/revision/audit.json",
                 "output/revision/excluded_vs_included.csv", "output/revision/revision_digest.json")


@pytest.fixture
def ca():
    return json.loads((REV / "cohort_accounting.json").read_text())


@pytest.fixture
def audit():
    return json.loads((REV / "audit.json").read_text())


@pytest.fixture
def evi():
    return pd.read_csv(REV / "excluded_vs_included.csv")


@pytest.fixture
def frozen_aim12():
    return json.loads((REV / "revision_digest.json").read_text())["frozen_aim12"]


@pytest.mark.requires_study_files(*OUTSIDE_FILES)
def test_outside_stays_add_up(ca, audit):
    assert ca["outside_n"] == ca["no_verbal_n"] + ca["aim3_excluded_n"] == 80
    assert ca["outside_deaths_n"] == ca["no_verbal_deaths_n"] + ca["aim3_excl_deaths_n"]
    assert ca["eligible_deaths_n"] == audit["sel_deaths_total"] + ca["outside_deaths_n"]
    assert ca["outside_deaths_pct"] == pytest.approx(100 * ca["outside_deaths_n"] / ca["eligible_deaths_n"])


@pytest.mark.requires_study_files(*OUTSIDE_FILES)
def test_deaths_column_matches_mortality(evi):
    assert (evi.deaths == (evi.n * evi.mortality_pct / 100).round().astype(int)).all()
    assert evi.set_index(["comparison", "group"]).loc[("audit_subset", "included"), "deaths"] == 2702


@pytest.mark.requires_study_files(*OUTSIDE_FILES)
def test_burden_bounds(ca, frozen_aim12):
    assert round(100 * ca["ever_nonassess_n"] / ca["with_verbal_n"], 1) == round(frozen_aim12["aim1.overall_pct_ever_nonassess"], 1)
    assert ca["burden_bound_lo_pct"] == pytest.approx(100 * ca["ever_nonassess_n"] / ca["neuro_first_stay_n"])
    assert ca["burden_bound_hi_pct"] == pytest.approx(100 * (ca["ever_nonassess_n"] + ca["no_verbal_n"]) / ca["neuro_first_stay_n"])


@pytest.mark.requires_study_files(*OUTSIDE_FILES)
def test_complete_case_share_with_outside_stays(ca, audit):
    v = 100 * (audit["sel_deaths_excluded"] + ca["outside_deaths_n"]) / (audit["sel_deaths_total"] + ca["outside_deaths_n"])
    assert ca["cc_pct_deaths_excluded_with_outside"] == pytest.approx(v)
    assert ca["cc_pct_deaths_excluded_with_outside"] > audit["sel_pct_deaths_excluded"]
