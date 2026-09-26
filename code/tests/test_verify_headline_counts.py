import json

import pytest

import verify_headline_counts as V
from config import REV

pytestmark = pytest.mark.requires_study_files("output/revision/revision_digest.json")


def _digest():
    return json.loads((REV / "revision_digest.json").read_text())


def test_the_regenerated_digest_matches_the_headline_counts():
    assert V.check(_digest()) == []


def test_a_moved_count_is_reported():
    d = _digest()
    d["audit"]["sel_deaths_excluded"] = 1965
    assert V.check(d) == ["audit:sel_deaths_excluded = 1965, expected 1966"]


def test_a_missing_key_is_reported():
    d = _digest()
    del d["feasible"]["official_n"]
    assert V.check(d) == ["feasible:official_n = None, expected 14215"]
