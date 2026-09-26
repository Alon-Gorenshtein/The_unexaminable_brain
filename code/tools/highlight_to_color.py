"""Convert highlighted runs to dark red text (the journal asks for changes in colored text).

    python3 code/tools/highlight_to_color.py FILE.docx [...]

Works on every run in the document body, including runs inside hyperlinks and table cells: the
highlight is removed and the run's text color is set to C00000. Prints the number of runs recolored.
"""
import sys

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

COLOR = "C00000"


def recolor(path):
    doc = Document(path)
    n = 0
    for r in doc.element.body.iter(qn("w:r")):
        rpr = r.find(qn("w:rPr"))
        hl = rpr.find(qn("w:highlight")) if rpr is not None else None
        if hl is None:
            continue
        rpr.remove(hl)
        color = rpr.find(qn("w:color"))
        if color is None:
            color = OxmlElement("w:color")
            # w:color precedes w:sz, w:szCs, w:highlight, w:u ... in the schema order of rPr
            anchor = next((rpr.find(qn(t)) for t in ("w:spacing", "w:w", "w:kern", "w:position", "w:sz",
                                                     "w:szCs", "w:u", "w:effect", "w:bdr", "w:shd", "w:fitText",
                                                     "w:vertAlign", "w:rtl", "w:cs", "w:em", "w:lang",
                                                     "w:eastAsianLayout") if rpr.find(qn(t)) is not None), None)
            if anchor is not None:
                anchor.addprevious(color)
            else:
                rpr.append(color)
        for a in list(color.attrib):
            del color.attrib[a]
        color.set(qn("w:val"), COLOR)
        n += 1
    doc.save(path)
    return n


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(f, recolor(f))
