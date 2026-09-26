import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT.parent / "_pub_assets" / "make_git_upload.py"
# docs/ is not part of the public copy, so every test here skips there.
pytestmark = pytest.mark.requires_study_files("docs/regenerate_git_upload.py")

# The "Reproducing the analysis" section as the shared builder's template writes it for this study.
TEMPLATE_README = """# T

## Reproducing the analysis

1. Clone this repository: `git clone https://github.com/Alon-Gorenshtein/The_unexaminable_brain.git`
2. Use Python 3.9 or later. Install the scientific stack: `pandas numpy scipy scikit-learn statsmodels` (and `lifelines`, `torch` where the scripts require them).
2. Point the data-path variable at the top of `code/config.py` to your local copy of MIMIC-IV.
3. Run the scripts in `code/` in numeric order (`00_*`, `01_*`, ...). Intermediate and final outputs are written to `./output/`.

## Repository contents
"""


def _wrapper():
    spec = importlib.util.spec_from_file_location("regenerate_git_upload", ROOT / "docs" / "regenerate_git_upload.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_patch_turns_the_template_steps_into_the_ones_that_point_to_the_code_readme():
    out = _wrapper().patch(TEMPLATE_README)
    steps = [ln for ln in out.splitlines() if ln[:2] in ("2.", "3.", "4.")]
    assert [s[:2] for s in steps] == ["2.", "3.", "4."]
    assert steps[0].count("`code/README.md`") == 1 and "script 60 needs its own environment" in steps[0]
    assert "eICU-CRD" in steps[1]
    assert steps[2].startswith("4. Run the scripts in the order given in `code/README.md`, not in plain numeric order")
    assert "Install the scientific stack" not in out and "in numeric order (`00_*`" not in out
    assert out.startswith("# T\n\n## Reproducing the analysis\n\n1. Clone") and out.endswith("## Repository contents\n")


def test_patch_refuses_text_without_the_expected_block():
    w = _wrapper()
    for bad in ("# T\n\n1. Clone\n2. something else\n",                                    # block absent
                TEMPLATE_README.replace("in numeric order", "in order"),                   # template wording changed
                TEMPLATE_README + w.TEMPLATE_STEPS):                                       # block twice
        with pytest.raises(ValueError, match="template steps 2-3"):
            w.patch(bad)


def test_patch_is_idempotent_and_only_touches_the_block():
    w = _wrapper()
    once = w.patch(TEMPLATE_README)
    assert w.patch(once) == once
    assert once.replace(w.STEPS, w.TEMPLATE_STEPS) == TEMPLATE_README


@pytest.mark.requires_study_files(str(BUILDER))
def test_the_wrappers_template_block_is_still_in_the_shared_builder():
    src = BUILDER.read_text(encoding="utf-8")
    for fragment in ("Install the scientific stack", "Point the data-path variable at the top of `code/config.py`",
                     "in numeric order (`00_*`, `01_*`, ...)"):
        assert fragment in src, fragment
    assert _wrapper().TEMPLATE_STEPS.count("Install the scientific stack") == 1


@pytest.mark.requires_study_files("git_upload/README.md")
def test_the_committed_public_readme_keeps_the_steps_and_not_the_old_instruction():
    text = (ROOT / "git_upload" / "README.md").read_text(encoding="utf-8")
    w = _wrapper()
    assert text.count(w.STEPS) == 1
    assert w.TEMPLATE_STEPS not in text and "Install the scientific stack" not in text
    assert "in numeric order (`00_*`" not in text
