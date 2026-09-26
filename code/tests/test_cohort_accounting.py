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
