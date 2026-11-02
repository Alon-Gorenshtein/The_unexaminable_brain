import re
from pathlib import Path

import pytest
from docx import Document
from docx.oxml.ns import qn

import check_packet as C
import set_table_widths as T

ROOT = Path(__file__).resolve().parents[2]
KEY = ("Phenotype", "n", "Ventilated (%)")
WIDTHS = {KEY: (3000, 2000, 4360)}


def _doc(path, header=KEY, widths=(1000, 1000, 1000)):
    d = Document()
    d.add_paragraph("Before the table")
    t = d.add_table(rows=2, cols=len(header))
    for c, h in zip(t.rows[0].cells, header):
        c.text = h
    for c, v in zip(t.rows[1].cells, ("Comparator", "70,705", "33.2")):
        c.text = v
    for g, w in zip(t._tbl.tblGrid.gridCol_lst, widths):
        g.set(qn("w:w"), str(w))
    d.save(path)


def _grid(path):
    return [[int(g.get(qn("w:w"))) for g in t._tbl.tblGrid.gridCol_lst] for t in Document(path).tables]


def _text(path):
    d = Document(path)
    return [p.text for p in d.paragraphs], [[c.text for c in r.cells] for t in d.tables for r in t.rows]


def test_apply_sets_the_declared_widths_and_leaves_every_word_alone(tmp_path):
    p = tmp_path / "a.docx"
    _doc(p)
    before = _text(p)
    assert T.apply(p, WIDTHS) == 1
    assert _grid(p) == [[3000, 2000, 4360]]
    assert _text(p) == before


def test_apply_is_idempotent(tmp_path):
    p = tmp_path / "a.docx"
    _doc(p)
    T.apply(p, WIDTHS)
    T.apply(p, WIDTHS)
    assert _grid(p) == [[3000, 2000, 4360]]


def test_apply_refuses_a_file_without_the_declared_table(tmp_path):
    p = tmp_path / "a.docx"
    _doc(p, header=("Rule", "AUROC", "Brier"))
    with pytest.raises(ValueError, match="not found"):
        T.apply(p, WIDTHS)


def test_apply_refuses_a_column_count_that_differs(tmp_path):
    p = tmp_path / "a.docx"
    _doc(p)
    with pytest.raises(ValueError, match="columns"):
        T.apply(p, {KEY: (5000, 4360)})


def test_check_fails_on_the_unwidened_file_and_passes_after_apply(tmp_path):
    p = tmp_path / "a.docx"
    _doc(p)
    assert any("differ from the declared" in m for m in T.check(p, WIDTHS))
    T.apply(p, WIDTHS)
    assert T.check(p, WIDTHS) == []


def test_check_names_a_missing_table_and_a_wrong_sum(tmp_path):
    p = tmp_path / "a.docx"
    _doc(p, header=("Rule", "AUROC", "Brier"))
    assert any("not found" in m for m in T.check(p, WIDTHS))
    assert any("sum to" in m for m in T.check(p, {KEY: (1, 2, 3)}))


def test_packet_guard_fails_on_the_narrow_grid_and_passes_after_the_tool(tmp_path):
    d = Document()
    for key, widths in T.TABLE_WIDTHS.items():
        t = d.add_table(rows=1, cols=len(key))
        for c, h in zip(t.rows[0].cells, key):
            c.text = h
        for g in t._tbl.tblGrid.gridCol_lst:
            g.set(qn("w:w"), "808")
        d.add_paragraph("")
    p = tmp_path / "m.docx"
    d.save(p)
    assert len(C.table_widths_set(p)) == len(T.TABLE_WIDTHS)
    T.apply(p)
    assert C.table_widths_set(p) == []


def test_declared_widths_fill_the_text_width_and_match_their_column_counts():
    for key, widths in T.TABLE_WIDTHS.items():
        assert sum(widths) == T.TEXT_WIDTH and len(widths) == len(key) and all(w > 0 for w in widths)


@pytest.mark.requires_study_files("manuscript/manuscript.md")
def test_declared_tables_are_the_header_rows_of_the_manuscript_source():
    rows = [tuple(c.strip() for c in line.strip().strip("|").split("|"))
            for line in (ROOT / "manuscript" / "manuscript.md").read_text(encoding="utf-8").splitlines()
            if line.startswith("|")]
    for key in T.TABLE_WIDTHS:
        assert key in rows, key[0]
