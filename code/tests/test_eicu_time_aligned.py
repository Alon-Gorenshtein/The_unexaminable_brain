import json
import pytest

from config import REV

pytestmark = pytest.mark.requires_study_files(
    "output/revision/eicu_time_aligned.json",
    "output/revision/eicu_probe.json")


def _load():
    return (json.loads((REV / "eicu_time_aligned.json").read_text()),
            json.loads((REV / "eicu_probe.json").read_text()))


def test_post_window_matches_the_probe_count():
    j, p = _load()
    assert j["all_w0_6h_n"] == p["n_stays_gcs_in_first_6h_of_vent"]
    assert p["n_stays_gcs_in_first_6h_of_vent"] >= p["gate_min_stays"] and p["decision"] == "GO"


def test_nearest_pre_window_is_separated_from_post_by_confidence_intervals():
    j, _ = _load()
    assert j["all_pre_6h_n"] > 0
    assert j["all_pre_6h_ge12_lo"] > j["all_w0_6h_ge12_hi"]      # nearest pre record vs nearest post record
    assert j["all_pre_6h_earliest_ge12_pct"] > j["all_pre_6h_ge12_pct"]   # the furthest record overstates the contrast


def test_paired_counts_are_consistent():
    j, _ = _load()
    for sub in ("all", "neuro"):
        assert j[f"{sub}_paired_n"] > 0
        both = j[f"{sub}_paired_pre_ge12_n"] - j[f"{sub}_paired_pre_only_ge12_n"]
        assert both == j[f"{sub}_paired_post_ge12_n"] - j[f"{sub}_paired_post_only_ge12_n"]
        assert both >= 0 and j[f"{sub}_paired_n"] <= j[f"{sub}_pre_6h_n"]
