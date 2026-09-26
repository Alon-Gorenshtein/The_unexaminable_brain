import pandas as pd
import pytest

from config import REV

pytestmark = pytest.mark.requires_study_files(
    "output/revision/time_to_first_non_assessable_by_phenotype.csv")


def test_quoted_median_times_match_the_table():
    t = pd.read_csv(REV / "time_to_first_non_assessable_by_phenotype.csv").set_index("phenotype")
    quoted = {"ICH": 0.9, "TBI": 1.0, "SAH": 1.1, "anoxic": 1.3, "SDH": 1.7, "AIS": 2.7, "comparator": 3.0}
    for ph, h in quoted.items():
        assert round(t.loc[ph, "median_h"], 1) == h, ph
    assert (t.n_stays > 0).all()
