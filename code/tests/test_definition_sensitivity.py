import json

import pandas as pd
import pytest

from config import PHENOTYPE_PRIORITY, REV, phenotype_masks, specific_phenotype_masks
from lib_cohort import NEURO, assign_hier, phenotype_flags, specific_labels

CASES = [("3481", 9, "anoxic", True), ("3485", 9, "anoxic", False), ("G931", 10, "anoxic", True),
         ("G9382", 10, "anoxic", False), ("43311", 9, "AIS", True), ("43310", 9, "AIS", False),
         ("43491", 9, "AIS", True), ("43490", 9, "AIS", False), ("I639", 10, "AIS", True),
         ("85201", 9, "TBI", True), ("80100", 9, "TBI", False), ("8020", 9, "TBI", False),
         ("80110", 9, "TBI", True), ("80160", 9, "TBI", True), ("80150", 9, "TBI", False),
         ("80240", 9, "TBI", False), ("S065X0A", 10, "TBI", True), ("4321", 9, "SDH", True), ("4320", 9, "SDH", False),
         ("I6200", 10, "SDH", True), ("I621", 10, "SDH", False), ("430", 9, "SAH", True), ("I61", 10, "ICH", True),
         ("3481", 10, "anoxic", False)]


@pytest.mark.parametrize("code,ver,ph,want", CASES)
def test_specific_masks(code, ver, ph, want):
    df = pd.DataFrame({"icd_code": [code], "icd_version": [ver]})
    assert bool(specific_phenotype_masks(df)[ph].iloc[0]) is want


def test_specific_masks_never_add_membership():
    # every code flagged specific is also flagged by the primary masks for the same phenotype
    nine = [p + s for p in ("348", "430", "431", "432", "433", "434", "800", "801", "802", "850", "851", "852",
                            "853", "854", "855") for s in ("", "0", "1", "5", "00", "01", "10", "11", "91")]
    ten = ["G931", "G9381", "G9382", "G935", "I60", "I609", "I61", "I619", "I620", "I6200", "I621", "I629", "I63",
           "I639", "I6320", "S06", "S065X0A", "S0650XA", "S0690XA", "S0100XA", "G40"]
    df = pd.DataFrame({"icd_code": nine + ten, "icd_version": [9] * len(nine) + [10] * len(ten)})
    prim, spec = phenotype_masks(df), specific_phenotype_masks(df)
    assert set(spec) == set(prim) == set(NEURO)
    for ph in NEURO:
        assert not (spec[ph] & ~prim[ph]).any(), ph
    assert (spec["SAH"] == prim["SAH"]).all() and (spec["ICH"] == prim["ICH"]).all()
    assert all(spec[ph].sum() < prim[ph].sum() for ph in ("SDH", "AIS", "TBI", "anoxic"))


DX = pd.DataFrame({"hadm_id": [1, 1, 2, 2, 3], "icd_code": ["3485", "3481", "80100", "I639", "43310"],
                   "icd_version": [9, 9, 9, 10, 9]})
IDS = [1, 2, 3]


def test_stay_with_specific_and_broad_code_is_kept():
    # hadm 1 carries 3485 and 3481 -> "anoxic"
    assert specific_labels(DX, IDS).loc[1] == "anoxic"
    assert assign_hier(phenotype_flags(DX, IDS), PHENOTYPE_PRIORITY).loc[1] == "anoxic"


def test_nonspecific_label_relabels_to_specific_phenotype():
    # hadm 2 carries 80100 and I639: primary label TBI, specific label AIS
    assert assign_hier(phenotype_flags(DX, IDS), PHENOTYPE_PRIORITY).loc[2] == "TBI"
    assert specific_labels(DX, IDS).loc[2] == "AIS"


def test_only_nonspecific_code_gives_nan():
    # hadm 3 carries 43310 only -> NaN
    assert assign_hier(phenotype_flags(DX, IDS), PHENOTYPE_PRIORITY).loc[3] == "AIS"
    assert pd.isna(specific_labels(DX, IDS).loc[3])
    assert list(specific_labels(DX, IDS).index) == IDS


FILES = ("output/revision/definition_sensitivity.json", "output/revision/feasible.json", "output/revision/audit.json")


@pytest.mark.requires_study_files(*FILES)
def test_primary_column_matches_registered_outputs():
    # primary_official_n == feasible official_n; primary_default_15_n == 3139; primary_audit_n == audit sel_n_total;
    # primary_auroc_official and primary_auroc_brennan within 0.0005 of audit.json
    j, f, a = (json.loads((REV / n).read_text()) for n in ("definition_sensitivity.json", "feasible.json", "audit.json"))
    assert j["primary_official_n"] == f["official_n"]
    assert j["primary_default_15_n"] == f["official_default_15_n"] == 3139
    assert j["primary_audit_n"] == a["sel_n_total"]
    assert j["primary_auroc_official"] == pytest.approx(a["official_lowest_em_auroc"], abs=0.0005)
    assert j["primary_auroc_brennan"] == pytest.approx(a["brennan_lowest_em_auroc"], abs=0.0005)


@pytest.mark.requires_study_files("output/revision/definition_sensitivity.json")
def test_specific_values_are_the_expected_ones():
    # removed_n == 1607; removed_TBI_n == 209; removed_AIS_n == 733; removed_anoxic_n == 567; removed_SDH_n == 98;
    # specific_analytic_n == 12628; specific_audit_n == 12597; specific_audit_deaths == 2551;
    # round(specific_ever_anoxic_pct, 1) == 91.8
    j = json.loads((REV / "definition_sensitivity.json").read_text())
    assert (j["removed_n"], j["removed_TBI_n"], j["removed_AIS_n"], j["removed_anoxic_n"], j["removed_SDH_n"]) == \
        (1607, 209, 733, 567, 98)
    assert (j["specific_analytic_n"], j["specific_audit_n"], j["specific_audit_deaths"]) == (12628, 12597, 2551)
    assert round(j["specific_ever_anoxic_pct"], 1) == 91.8


@pytest.mark.requires_study_files("output/revision/definition_sensitivity.json")
def test_removed_stays_by_icd_version_and_anoxic_characteristics():
    # the removed stays split by the ICD version of their codes; the shares are of all eligible stays in that version
    j = json.loads((REV / "definition_sensitivity.json").read_text())
    assert 0 < j["removed_icd9_n"] <= j["removed_n"]
    removed_icd10 = j["removed_icd10_n"]
    assert j["removed_icd9_n"] + removed_icd10 == j["removed_n"]
    assert j["removed_icd9_pct"] > j["removed_icd10_pct"] > 0
    # eligible stays in ICD-9 = removed_icd9_n / pct, in ICD-10 = removed_icd10 / pct; together all eligible stays
    n9, n10 = j["removed_icd9_n"] / (j["removed_icd9_pct"] / 100), removed_icd10 / (j["removed_icd10_pct"] / 100)
    assert n9 + n10 == pytest.approx(j["primary_eligible_n"], abs=1)
