import pathlib
import subprocess

import pytest

import check_response_quotes as crq

ROOT = pathlib.Path(__file__).resolve().parents[2]

ORIG = ("The default rule was applied to every stay. Complete-case handling excluded 3530 of 12,404 "
        "patients (28.5%) and the AUROC was 0.746 (0.732–0.759).")
REV = ("A verbal entry of \"No Response-ETT\" is non-assessable. The eICU-CRD data dictionary flags a GCS "
       "that could not be scored^19^, and the table is below.\n\n| Brennan estimate | 0.820 (0.812 to 0.828) |")


def _letter(kind, *quote_lines, heading="### Comment 1.1."):
    return "\n".join([heading, "", "Response. Thank you.", "", f"**{kind}:**", ""]
                     + [f"> {q}" for q in quote_lines]) + "\n"


def test_passing_tiny_example():
    text = _letter("Original", "Complete-case handling excluded 3530 of 12,404 patients (28.5%)") + \
        _letter("Revised", "A verbal entry of \"No Response-ETT\" is non-assessable.", heading="### Comment 1.2.")
    n, fails = crq.check(text, [ORIG], [REV])
    assert fails == [] and n == 2


def test_absent_fragment_fails_with_comment_number():
    text = _letter("Revised", "A verbal entry of \"No Response-ETT\" is always scored as 1.", heading="### Comment 1.7.")
    n, fails = crq.check(text, [ORIG], [REV])
    assert len(fails) == 1 and "Comment 1.7" in fails[0] and "always scored" in fails[0]


def test_original_is_not_checked_against_the_revised_text():
    text = _letter("Original", "A verbal entry of \"No Response-ETT\" is non-assessable.")
    assert crq.check(text, [ORIG], [REV])[1]


def test_ellipsis_joins_fragments_of_one_paragraph():
    ok = _letter("Original", "The default rule was applied ... excluded 3530 of 12,404 patients … was 0.746")
    assert crq.check(ok, [ORIG], [REV])[1] == []
    bad = _letter("Original", "The default rule was applied ... excluded 9999 of 12,404 patients")
    fails = crq.check(bad, [ORIG], [REV])[1]
    assert len(fails) == 1 and "9999" in fails[0]


def test_typography_is_normalised():
    # curly quotes, a non-breaking space, a superscript citation that the quote omits, bold markers,
    # and an en dash in the source matched by a hyphen in the quote
    text = _letter("Revised", "A verbal entry of “No Response-ETT” is **non-assessable.** The eICU-CRD "
                   "data dictionary flags a GCS that could not be scored, and")
    assert crq.check(text, [ORIG], [REV])[1] == []
    text2 = _letter("Original", "the AUROC was 0.746 (0.732-0.759).")
    assert crq.check(text2, [ORIG], [REV])[1] == []


def test_multi_paragraph_and_table_rows_are_checked_separately():
    text = _letter("Revised", "A verbal entry of \"No Response-ETT\" is non-assessable.", "",
                   "| Brennan estimate | 0.820 (0.812 to 0.828) |").replace("> \n", ">\n")
    n, fails = crq.check(text, [ORIG], [REV])
    assert fails == [] and n == 2
    bad = text.replace("0.812 to 0.828", "0.771 to 0.796")
    assert len(crq.check(bad, [ORIG], [REV])[1]) == 1


def test_label_without_block_quote_and_empty_letter_fail():
    text = "### Comment 1.4.\n\n**Revised:**\n\nThe text was changed.\n"
    fails = crq.check(text, [ORIG], [REV])[1]
    assert any("not followed by a block quote" in f for f in fails)
    assert crq.check("No quotations here.\n", [ORIG], [REV])[1]


def test_revised_quote_found_only_in_the_submitted_text_fails():
    text = _letter("Revised", "Complete-case handling excluded 3530 of 12,404 patients (28.5%)")
    fails = crq.check(text, [ORIG], [REV])[1]
    assert len(fails) == 1 and "[Revised]" in fails[0]


def test_each_kind_accepts_either_file_of_its_version():
    supp_orig, supp_rev = "Submitted supplement sentence.", "Revised supplement sentence."
    text = _letter("Original", "Submitted supplement sentence.") + \
        _letter("Revised", "Revised supplement sentence.", heading="### Comment 1.2.")
    assert crq.check(text, [ORIG, supp_orig], [REV, supp_rev])[1] == []


def test_only_a_comment_heading_changes_the_attribution():
    text = ("### Comment 1.3.\n\n**See also Comment 1.9 and Comment 1.11.** More prose on Comment 1.12.\n\n"
            "**Revised:**\n\n> this sentence is not in either file\n\n"
            "## Additional corrections\n\nAs noted under Comment 1.5, ...\n\n"
            "**Original:**\n\n> nor is this one\n")
    fails = crq.check(text, [ORIG], [REV])[1]
    assert fails[0].startswith("Comment 1.3 [Revised]"), fails
    assert fails[1].startswith("Additional corrections [Original]"), fails


@pytest.mark.skipif(not (ROOT / "manuscript" / "response_to_reviewers.md").exists(), reason="no response yet")
def test_real_response_passes():
    r = subprocess.run(["python3", str(ROOT / "code" / "tools" / "check_response_quotes.py")], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


SRC_TABLE = ("Intro.\n\n| Rule | AUROC (95% CI) | Brier |\n|---|---|---:|\n"
             "| Official first-day GCS | 0.749 (0.739 to 0.760) | 0.135 |\n"
             "| Brennan estimate | 0.820 (0.812 to 0.828) | 0.121 |\n\nAfter.\n")


def _table_letter(rows, kind="Revised"):
    return "\n".join(["### Comment 1.7.", "", "Response. Thanks.", "", f"**{kind}:**", "",
                      "| Rule | AUROC (95% CI) | Brier |", "|---|---|---:|", *rows,
                      "", "> Panel note from the source.", "", "Prose after the quotation."]) + "\n"


def test_table_quote_passes_cell_by_cell():
    text = _table_letter(["| Brennan estimate | 0.820 (0.812 to 0.828) | **0.121** |"])
    n, fails = crq.check(text, [ORIG], [SRC_TABLE + "\nPanel note from the source.\n"])
    assert fails == [] and n == 2  # one table row and the paragraph after it


def test_table_quote_with_one_changed_cell_fails():
    text = _table_letter(["| Brennan estimate | 0.820 (0.812 to 0.828) | 0.122 |"])
    n, fails = crq.check(text, [ORIG], [SRC_TABLE + "\nPanel note from the source.\n"])
    assert len(fails) == 1 and "0.122" in fails[0] and "Comment 1.7" in fails[0]


def test_table_quote_with_a_foreign_header_fails():
    text = _table_letter(["| Brennan estimate | 0.820 (0.812 to 0.828) | 0.121 |"]).replace(
        "| Rule | AUROC", "| Strategy | AUROC")
    assert crq.check(text, [ORIG], [SRC_TABLE + "\nPanel note from the source.\n"])[1]


def test_table_quote_is_checked_against_its_own_version():
    text = _table_letter(["| Brennan estimate | 0.820 (0.812 to 0.828) | 0.121 |"], kind="Original")
    assert crq.check(text, [ORIG], [SRC_TABLE + "\nPanel note from the source.\n"])[1]


def test_real_response_table_quote_breaks_when_a_cell_changes():
    resp = ROOT / "manuscript" / "response_to_reviewers.md"
    if not resp.exists():
        pytest.skip("no response yet")
    text = resp.read_text(encoding="utf-8")
    row = "| Default-to-15 | 0.744 (0.734 to 0.755) | 0.135 | 13.76 | 1.23 |"
    assert row in text
    rev = [(ROOT / "manuscript" / f).read_text(encoding="utf-8") for f in ("manuscript.md", "supplement.md")]
    bad = text.replace(row, row.replace("13.76", "13.67"))
    fails = crq.check(bad, [], rev)[1]
    assert any("13.67" in f for f in fails)
