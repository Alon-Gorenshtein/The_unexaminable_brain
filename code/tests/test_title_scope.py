import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MS = ROOT / "manuscript"
OLD = "Mechanically Ventilated Patients With Acute Brain Injury"


@pytest.mark.requires_study_files("manuscript/title.txt", "manuscript/manuscript.md", "manuscript/supplement.md", "manuscript/cover_letter_revision.md")
def test_one_title_in_every_source_and_no_ventilation_scope():
    title = (MS / "title.txt").read_text(encoding="utf-8").strip()
    assert "Mechanically Ventilated" not in title and "Critically Ill Adults" in title
    for name in ("manuscript.md", "supplement.md", "cover_letter_revision.md"):
        assert title in (MS / name).read_text(encoding="utf-8"), name
    for name in ("manuscript.md", "supplement.md", "cover_letter_revision.md"):
        assert OLD not in (MS / name).read_text(encoding="utf-8"), name


@pytest.mark.requires_study_files("manuscript/response_to_reviewers.md")
def test_the_response_shows_the_old_title_only_as_a_quotation():
    for line in (MS / "response_to_reviewers.md").read_text(encoding="utf-8").splitlines():
        if OLD in line:
            assert line.lstrip().startswith(">") or line.startswith("**Original"), line[:120]


DECL = ["Acknowledgments", "Author Contributions", "Statements and Declarations", "Ethical considerations",
        "Consent to participate", "Consent for publication", "Declaration of conflicting interest", "Funding statement",
        "Data availability"]
TOP = DECL[:3]
AUTHORS = ["Alon Gorenshtein", "Yosef Adiniaev", "Mahmud Omar", "Yiftach Barash", "Eyal Klang", "Oved Daniel"]


def declaration_problems(t):
    """Problems with the block from Acknowledgments to References: every required heading present and in order
    before the References, no bracketed placeholder left, and each author named under Author Contributions."""
    pos = [t.find(f"\n{'##' if h in TOP else '###'} {h}\n") for h in DECL]
    refs = t.find("\n## References\n")
    if not all(p > 0 for p in pos) or refs < 0:
        return [f"missing heading {h!r}" for h, p in zip(DECL, pos) if p <= 0] + ([] if refs >= 0 else ["no References"])
    problems = [] if pos == sorted(pos) and pos[-1] < refs else ["headings out of order"]
    problems += [f"bracket {b!r}" for b in re.findall(r"\[[^\]]*\]", t[pos[0]:refs])]
    start = pos[1] + len("\n## Author Contributions\n")
    contrib = t[start:t.find("\n## ", start)]
    problems += [f"no contribution line for {a}" for a in AUTHORS if f"- **{a}:** " not in contrib]
    return problems


@pytest.mark.requires_study_files("manuscript/manuscript.md")
def test_journal_required_headings_exist_in_order_before_the_references():
    assert declaration_problems((MS / "manuscript.md").read_text(encoding="utf-8")) == []


GOOD = ("\n## Conclusion\n\nText.\n\n## Acknowledgments\n\nText.\n\n## Author Contributions\n\n"
        + "".join(f"- **{a}:** Writing - review and editing.\n" for a in AUTHORS)
        + "\n## Statements and Declarations\n\n"
        + "".join(f"### {h}\n\nText.\n\n" for h in DECL[3:]) + "## References\n\n1. Ref.\n")


def test_declaration_guard_passes_a_complete_block():
    assert declaration_problems(GOOD) == []


def test_declaration_guard_catches_a_placeholder_bracket():
    assert declaration_problems(GOOD.replace("Text.\n\n## Author", "[to be added]\n\n## Author")) == [
        "bracket '[to be added]'"]


def test_declaration_guard_catches_the_old_credit_marker():
    bad = GOOD.replace("- **Oved Daniel:** Writing - review and editing.\n", "[CRediT roles to be added by the authors]\n")
    assert declaration_problems(bad) == ["bracket '[CRediT roles to be added by the authors]'",
                                         "no contribution line for Oved Daniel"]


def test_declaration_guard_wants_acknowledgments_before_author_contributions():
    ack = "## Acknowledgments\n\nText.\n\n"
    bad = GOOD.replace(ack, "").replace("## Statements and Declarations", ack + "## Statements and Declarations")
    assert declaration_problems(bad) == ["headings out of order"]


def test_declaration_guard_wants_the_acknowledgments_heading():
    assert declaration_problems(GOOD.replace("## Acknowledgments\n", "")) == ["missing heading 'Acknowledgments'"]


def test_declaration_guard_wants_every_author_under_author_contributions():
    bad = GOOD.replace("- **Mahmud Omar:** Writing - review and editing.\n", "")
    bad = bad.replace("### Funding statement\n\nText.", "### Funding statement\n\n- **Mahmud Omar:** Text.")
    assert declaration_problems(bad) == ["no contribution line for Mahmud Omar"]
