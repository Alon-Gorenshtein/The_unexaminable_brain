import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

STUDY_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _exists(p):
    p = pathlib.Path(p)
    return (p if p.is_absolute() else STUDY_ROOT / p).exists()


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_study_files(*paths): skip when a listed file (relative to the study root, or absolute) is absent. "
        "These are analysis outputs derived from the credentialed data, the first-submission tables, or the "
        "manuscript and packet sources; they exist in the study repository but not in the public code copy.")


def pytest_collection_modifyitems(config, items):
    for item in items:
        missing = [str(p) for m in item.iter_markers("requires_study_files") for p in m.args if not _exists(p)]
        if missing:
            item.add_marker(pytest.mark.skip(
                reason="needs " + ", ".join(sorted(set(missing))) + " (written by the analysis from the credentialed "
                       "data or part of the manuscript sources; not included in the public code copy)"))
