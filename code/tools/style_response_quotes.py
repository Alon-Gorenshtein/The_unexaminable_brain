"""Style the quotations of the response letter by their label.

    python3 code/tools/style_response_quotes.py response_to_reviewers.docx

Run after the house colorize_response_quotes.py. That script colors every block quote after a
"Response." paragraph blue and resets only at level-1 and level-2 headings, so in this letter, whose
comments are level-3 headings, the reviewer's own comments and the quotations of the submitted text
came out in the color reserved for revised text. Here each quotation is styled by the label that
precedes it. A quotation is a run of block quotes ("Block Text" paragraphs) and tables that follows
an "Original:" or "Revised:" label:

- after a "Revised:" label: dark blue (1F4E79), not italic (revised manuscript text);
- after an "Original:" label: italic, no color (submitted text);
- a block quote with no label (the reviewer's comment): italic, no color;
- a table with no label (the letter's own tables): left as it is.

A heading, a "Comment N.M." line, a "Response." paragraph, another label or any ordinary paragraph
ends the quotation. Prints the number of block quotes and tables styled each way."""
import re
import sys

import docx
from docx.shared import RGBColor

BLUE = RGBColor(0x1F, 0x4E, 0x79)


def _style_runs(runs, revised):
    for r in runs:
        if revised:
            r.font.color.rgb = BLUE
            r.italic = None
        else:
            r.font.color.rgb = None
            r.italic = True


def style(path):
    d = docx.Document(path)
    label = None
    n_blue = n_plain = n_tables = 0
    for block in d.iter_inner_content():
        if hasattr(block, "rows"):  # a table
            if label is None:
                continue
            for row in block.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        _style_runs(p.runs, label == "Revised")
            n_tables += 1
            continue
        p = block
        name = p.style.name if p.style is not None else ""
        text = p.text.strip()
        bold = bool(p.runs) and bool(p.runs[0].bold)
        if bold and re.match(r"(Original|Revised):?$", text):
            label = text.rstrip(":")
            continue
        if name == "Block Text":
            revised = label == "Revised"
            _style_runs(p.runs, revised)
            n_blue += revised
            n_plain += not revised
            continue
        if text or name.startswith("Heading"):
            label = None  # headings, Comment and Response lines and ordinary prose end a quotation
    d.save(path)
    print(f"{path}: {n_blue} revised-text quotes blue, {n_plain} reviewer or submitted-text quotes italic, "
          f"{n_tables} quoted tables styled")
    return n_blue, n_plain, n_tables


if __name__ == "__main__":
    for f in sys.argv[1:]:
        style(f)
