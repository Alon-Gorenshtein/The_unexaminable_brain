"""Set the column widths of the main-text tables whose words the shared width filter breaks.

    python3 code/tools/set_table_widths.py FILE.docx [...]

The shared filter sizes each column from its character count and the table then keeps those widths
(fixed layout), so a bold header word such as "Phenotype" or "Ventilated", the row label "Comparator"
and the count "70,705" can be wider than their column and break across lines ("Phenoty / pe",
"70,70 / 5"). TABLE_WIDTHS names the tables of this manuscript by the text of their header row and
gives each column's width in twips; the widths sum to the text width of the page (6.5 in). Only the
column grid changes; no cell text, number, run property or colour does, so the colored and the clean
manuscript stay identical in text.

The widths come from the longest unbreakable piece of each column (the text split at spaces, hyphens
and slashes) in 10 pt CMU Serif, with the bold header row measured bold, plus the 108-twip cell margin
on each side and a margin of about 8 percent. CMU Serif was measured on the first revision's PDF
preview (bold about 1.2 times Times New Roman Bold; digits equal), which is wider than the Times New
Roman that Word and LibreOffice substitute when CMU Serif is not installed.
"""
import sys

from docx import Document
from docx.oxml.ns import qn

TEXT_WIDTH = 9360  # twips: US Letter, 1 in margins

TABLE_WIDTHS = {
    # Table 1. Comparator was 961, so "Comparator" broke; "Anoxic" and "Overall" were within a few twips.
    ("Characteristic", "TBI", "ICH", "SAH", "SDH", "AIS", "Anoxic", "Comparator", "Overall neuro"):
        (1827, 759, 759, 770, 770, 759, 1017, 1625, 1074),
    # Table 2. Phenotype, n, Ventilated and Ventilated: were narrower than their longest word or number.
    ("Phenotype", "n", "Ventilated (%)", "Ever non-assessable (%)", "Ventilated: ever (%)",
     "Ventilated: median fraction (%)", "Ventilated: median fraction, first 72 h (%)"):
        (1399, 827, 1384, 1341, 1470, 1470, 1469),
}


def _header(table):
    return tuple(c.text for c in table.rows[0].cells)


def _grid(table):
    return [int(g.get(qn("w:w"))) for g in table._tbl.tblGrid.gridCol_lst]


def apply(path, widths=None):
    """Write the declared column widths into every matching table of the file; returns the number of
    tables set. A declared table that the file lacks raises, so a changed header cannot silently skip it."""
    widths = TABLE_WIDTHS if widths is None else widths
    doc = Document(path)
    done = set()
    for t in doc.tables:
        key = _header(t)
        if key in widths:
            cols = t._tbl.tblGrid.gridCol_lst
            if len(cols) != len(widths[key]):
                raise ValueError(f"{path}: table {key[0]!r} has {len(cols)} columns, {len(widths[key])} declared")
            for g, v in zip(cols, widths[key]):
                g.set(qn("w:w"), str(v))
            done.add(key)
    lost = [k[0] for k in widths if k not in done]
    if lost:
        raise ValueError(f"{path}: declared table(s) not found by header row: {lost}")
    doc.save(path)
    return len(done)


def check(path, widths=None):
    """Problems (empty means pass): a declared table is missing from the file, or its column grid is not
    the declared one, or the declared widths do not sum to the text width."""
    widths = TABLE_WIDTHS if widths is None else widths
    problems = []
    seen = {}
    for t in Document(path).tables:
        if _header(t) in widths:
            seen[_header(t)] = _grid(t)
    for key, want in widths.items():
        name = f"{path.name if hasattr(path, 'name') else path}: table {key[0]!r}"
        if sum(want) != TEXT_WIDTH:
            problems.append(f"{name}: declared widths sum to {sum(want)}, not {TEXT_WIDTH}")
        elif key not in seen:
            problems.append(f"{name}: not found by its header row")
        elif seen[key] != list(want):
            problems.append(f"{name}: column widths {seen[key]} differ from the declared {list(want)}")
    return problems


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(f"  {f}: column widths set in {apply(f)} tables")
