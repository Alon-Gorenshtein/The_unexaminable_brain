import pathlib
import subprocess
import sys

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CODE, TAB, V1 = ROOT / "code", ROOT / "output" / "tables", ROOT / "output" / "v1_submitted"

CASES = {
    "12_aim3_audit.py": ["table3_aim3_by_phenotype.csv", "table_selection_effect.csv",
                         "table_reclass_impute1_vs_kramer.csv"],
    "16_calibration.py": ["table_calibration.csv"],
    "17_derived_gcs_audit.py": ["table_derived_audit.csv", "table_sofa_crosstab.csv"],
    "18_selection_timecourse.py": ["table_selection_model.csv", "table_smd.csv",
                                   "table_no_assessable_timecourse.csv"],
}
NEEDS = ["output/intermediate/cohort.parquet", "output/intermediate/aim4_exams.parquet",
         "output/intermediate/chart_neuro.parquet", "output/intermediate/infusions.parquet",
         "output/intermediate/vent.parquet",
         *[f"output/v1_submitted/{n}" for names in CASES.values() for n in names]]


@pytest.mark.requires_study_files(*NEEDS)
@pytest.mark.parametrize("script", list(CASES))
def test_rerun_matches_submitted_tables(script):
    # The scripts write into the tracked output/tables folder. Keep the tracked bytes, delete the targets so a
    # script that stops writing one fails here, and put the tracked bytes back afterwards so a run of the suite
    # leaves the working tree unchanged.
    saved = {n: (TAB / n).read_bytes() for n in CASES[script] if (TAB / n).exists()}
    try:
        for n in CASES[script]:
            if (TAB / n).exists():
                (TAB / n).unlink()
        r = subprocess.run([sys.executable, script], cwd=CODE, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-800:]
        for name in CASES[script]:
            assert (TAB / name).exists(), f"{script} did not write {name}"
            pd.testing.assert_frame_equal(pd.read_csv(TAB / name), pd.read_csv(V1 / name),
                                          check_exact=False, rtol=1e-9, atol=1e-9)
    finally:
        for n, b in saved.items():
            (TAB / n).write_bytes(b)
