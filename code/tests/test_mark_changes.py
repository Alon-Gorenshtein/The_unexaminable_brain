import json
import subprocess
from pathlib import Path

import pytest

from mark_changes import mark, normalise, sentence_spans, strip_marks

ROOT = Path(__file__).resolve().parents[2]
PANDOC_FROM = "gfm+superscript+attributes+raw_attribute+bracketed_spans"

BASE = """# A Title

**Word count:** 100.

## Introduction

The scale has three parts^1,2^. It is charted by nurses. Sedation depresses all components^3^.

## Results

| Row | Value | Share |
|---|---:|---:|
| Alpha | 12 | 3.4 |
| Beta | 56 | 7.8 |

## References

1. Teasdale G, Jennett B. Assessment of coma. *Lancet*. 1974;2:81-84.
2. Brennan PM, Murray GD. A practical method. *J Neurosurg*. 2020;135:214-219.
"""


def _spans(md):
    return md.count("]{.mark}")


def _marked_units(md):
    """Content of each `[...]{.mark}` span (spans are never nested)."""
    out, stack = [], []
    for i, c in enumerate(md):
        if c == "[":
            stack.append(i)
        elif c == "]" and stack:
            s = stack.pop()
            if md.startswith("]{.mark}", i):
                out.append(md[s + 1:i])
    return out


def _table_shape(md):
    doc = json.loads(subprocess.run(["pandoc", "-f", PANDOC_FROM, "-t", "json"], input=md,
                                    capture_output=True, text=True, check=True).stdout)
    shapes = []
    for blk in doc["blocks"]:
        if blk["t"] == "Table":
            head, bodies = blk["c"][3], blk["c"][4]
            rows = head[1] + [r for b in bodies for r in b[3]]
            shapes.append((len(rows), [len(r[1]) for r in rows]))
    return shapes


def test_identical_text_has_no_spans():
    out, n = mark(BASE, BASE)
    assert n == 0 and out == BASE


def test_one_changed_sentence_gives_one_span_with_only_that_sentence():
    new = BASE.replace("It is charted by nurses.", "It is charted by nurses many times a day.")
    out, n = mark(BASE, new)
    assert n == 1 and _marked_units(out) == ["It is charted by nurses many times a day."]


def test_inserted_paragraph_marks_every_sentence():
    para = "We added this. It has two sentences^4^."
    new = BASE.replace("## Results\n", para + "\n\n## Results\n")
    out, n = mark(BASE, new)
    assert _marked_units(out) == ["We added this.", "It has two sentences^4^."]


def test_table_cell_change_marks_only_that_cell_and_table_still_parses():
    new = BASE.replace("| Beta | 56 | 7.8 |", "| Beta | 57 | 7.8 |")
    out, n = mark(BASE, new)
    assert _marked_units(out) == ["57"]
    assert "| Beta | [57]{.mark} | 7.8 |" in out
    assert _table_shape(out) == _table_shape(new) == [(3, [3, 3, 3])]


def test_value_moved_to_another_cell_is_still_a_change():
    new = BASE.replace("| Beta | 56 | 7.8 |", "| Beta | 12 | 7.8 |")
    assert _marked_units(mark(BASE, new)[0]) == ["12"]


def test_citation_only_renumbering_is_not_a_change():
    new = (BASE.replace("parts^1,2^", "parts^2,3^").replace("components^3^", "components^1^")
           .replace("1. Teasdale", "9. Teasdale"))
    assert mark(BASE, new)[1] == 0


def test_emphasis_and_quote_changes_are_not_changes():
    old = 'The rule was "default".\n'
    new = "The rule was **“default”**.\n"
    assert mark(old, new)[1] == 0


def test_moved_sentence_is_not_marked():
    new = BASE.replace("The scale has three parts^1,2^. It is charted by nurses.",
                       "It is charted by nurses. The scale has three parts^1,2^.")
    assert mark(BASE, new)[1] == 0


def test_deleted_text_produces_nothing():
    new = BASE.replace(" It is charted by nurses.", "")
    assert mark(BASE, new)[1] == 0


def test_changed_heading_and_reference_entry():
    new = BASE.replace("## Results", "## Main Results").replace("1974;2:81-84", "1974;2(7872):81-84")
    units = _marked_units(mark(BASE, new)[0])
    assert units == ["Main Results",
                     "Teasdale G, Jennett B. Assessment of coma. *Lancet*. 1974;2(7872):81-84."]


@pytest.mark.parametrize("text, n", [
    ("Rates differed (A vs. B). Next sentence.", 2),
    ("Some rules, e.g. Brennan, were used. Next.", 2),
    ("That is, i.e. None here. Next.", 2),
    ("As Brennan et al. Showed, it works. Next.", 2),
    ("The value was 0.5 in both. Next.", 2),
    ("See eTable 3. Then read S1.5 for details. Next.", 3),
    ("As shown in Fig. 2, it rose. Next.", 2),
    ("P < .001. Next.", 2),
    ("It ended^12^. Then another. And (a third). Fourth.", 4),
    ("**Table 1. Cohort by phenotype.** Values are n (%).", 2),
    ("lowercase after a period. starts no new sentence.", 1),
])
def test_sentence_splitting(text, n):
    spans = sentence_spans(text)
    assert len(spans) == n
    assert "".join(text[a:b] for a, b in spans).replace(" ", "") == text.replace(" ", "")


def test_normalise():
    assert normalise("**The** rule^1,2^ was  “x”") == 'The rule was "x"'


@pytest.mark.requires_study_files("manuscript/manuscript.md")
def test_round_trip_and_no_nesting_on_real_manuscript():
    old = subprocess.run(["git", "-C", str(ROOT), "show", "jicm-v1:manuscript/manuscript.md"],
                         capture_output=True, text=True)
    if old.returncode:
        pytest.skip("submitted version tag jicm-v1 not available")
    new = (ROOT / "manuscript" / "manuscript.md").read_text(encoding="utf-8")
    out, n = mark(old.stdout, new)
    assert n > 0 and strip_marks(out) == new
    assert all("{.mark}" not in u for u in _marked_units(out))  # no nested spans
    assert _table_shape(out) == _table_shape(new)
    # pandoc sees exactly one mark span per inserted wrapper, and the same words as the clean text
    native = subprocess.run(["pandoc", "-f", PANDOC_FROM, "-t", "native"], input=out,
                            capture_output=True, text=True, check=True).stdout
    assert native.count('[ "mark" ]') == n
    plain = [subprocess.run(["pandoc", "-f", PANDOC_FROM, "-t", "plain", "--wrap=none"], input=t,
                            capture_output=True, text=True, check=True).stdout for t in (out, new)]
    assert plain[0] == plain[1]
