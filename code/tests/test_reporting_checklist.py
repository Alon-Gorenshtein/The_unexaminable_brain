import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MS = ROOT / "manuscript"
CHECKLIST = MS / "reporting_checklist.md"


def _headings(md):
    return {m.group(1).strip().lower() for m in re.finditer(r"^#{1,4}\s+(.+)$", md, re.M)}


def _rows(text):
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and re.match(r"^\d+(\.\d+)?[a-e]?$", cells[0]):
            yield cells


def _unresolved(text, main_md, supp_md):
    """(item, part) for every "Where reported" entry that names nothing in the manuscript or supplement.
    Entries are separated by ";". "Not applicable: ..." and "Not reported: ..." carry a reason and are
    not looked up; "Title" and "Abstract" name the front of the paper; "Table N" and "Figure N" name a
    main-text caption; "Supplement X" names an eMethods heading, an eTable or eFigure caption, or a
    supplement heading; anything else is "Section" or "Section, Subsection", both of which must be
    manuscript headings."""
    main, supp = _headings(main_md), _headings(supp_md)
    bad = []
    for item, _, where in _rows(text):
        if not where:
            bad.append((item, ""))
        for part in (p.strip() for p in where.split(";")):
            low = part.lower()
            if low.startswith(("not applicable:", "not reported:")) or low in ("title", "abstract"):
                continue
            if re.match(r"^(table|figure) \d+$", low):
                ok = f"**{part}." in main_md
            elif low.startswith("supplement "):
                rest = part[len("supplement "):]
                if rest.startswith("eMethods "):
                    ok = any(h.startswith(rest[len("eMethods "):].lower() + " ") for h in supp)
                elif re.match(r"^e(Table|Figure) \d+$", rest):
                    ok = f"**{rest}." in supp_md
                else:
                    ok = rest.lower() in supp
            else:
                ok = all(s.strip().lower() in main for s in part.split(",", 1))
            if not ok:
                bad.append((item, part))
    return bad


@pytest.mark.requires_study_files("manuscript/reporting_checklist.md", "manuscript/manuscript.md", "manuscript/supplement.md")
def test_every_item_has_an_answer_and_every_named_section_exists():
    text = CHECKLIST.read_text(encoding="utf-8")
    items = [r[0] for r in _rows(text)]
    assert len(items) >= 35          # 22 STROBE + 13 RECORD
    assert len(items) == len(set(items))
    main_md = (MS / "manuscript.md").read_text(encoding="utf-8")
    supp_md = (MS / "supplement.md").read_text(encoding="utf-8")
    assert _unresolved(text, main_md, supp_md) == []


def test_resolver_rejects_names_that_do_not_exist():
    main_md = "## Methods\n\n### Cohort Identification\n\n**Table 1. Cohort.**\n"
    supp_md = "## S1.5 Model\n\n**eTable 3. Excluded.**\n"
    good = ("| 1a | x | Methods, Cohort Identification; Table 1; Supplement eMethods S1.5; "
            "Supplement eTable 3; Abstract; Not reported: none |\n")
    assert _unresolved(good, main_md, supp_md) == []
    for part in ("Methods, Cohort", "Results, Cohort Identification", "Table 2", "Supplement eMethods S1.6",
                 "Supplement eMethods S1", "Supplement eTable 4", "Supplement S9 Nothing", "Discussion"):
        assert _unresolved(f"| 1a | x | {part} |\n", main_md, supp_md) == [("1a", part)], part
    assert _unresolved("| 1a | x |  |\n", main_md, supp_md)


@pytest.mark.requires_study_files("manuscript/reporting_checklist.md", "manuscript/title.txt")
def test_starts_with_the_title_and_has_no_line_or_page_numbers_or_personal_names():
    t = CHECKLIST.read_text(encoding="utf-8")
    title = (MS / "title.txt").read_text(encoding="utf-8").strip()
    assert t.splitlines()[0] == f"# {title}"
    assert not re.search(r"\b(line|lines|page|pages|p\.)\s*\d", t, re.I)
    assert "Gorenshtein" not in t and "signature" not in t.lower()
    assert not re.search(r"\b20\d\d-\d\d-\d\d\b", t)
