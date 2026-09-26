import json
import pytest

from config import REV

pytestmark = pytest.mark.requires_study_files(
    "output/revision/feasible.json")


def test_decomposition_adds_up():
    j = json.loads((REV / "feasible.json").read_text())
    assert j["official_15_any_ett_n"] + j["official_15_no_ett_n"] == j["official_15_n"]
    assert j["official_default_15_impossible_n"] <= j["official_default_15_n"] <= j["official_15_n"]
    assert j["legacy_population_lowest_em_na_n"] == 3530
    assert j["legacy_population_lowest_em_na_impossible_n"] == 3438      # the submitted population
    assert j["lowest_em_na_n"] > 3530                                     # corrected population
    # independent check: the DuckDB carry-forward frame and the corrected exam table agree on the count
    assert abs(j["lowest_em_na_n"] - j["carryforward_lowest_em_na_n"]) <= 5
