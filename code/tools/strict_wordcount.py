"""Strict word count from a built .docx: Introduction through Conclusion, including headings,
table captions and table cells (tables sit after References in this manuscript). The counted
body ends at the first of Acknowledgments, Author Contributions, Statements and Declarations
or References.

The Introduction, References and Tables headings are all mandatory; a missing one raises
ValueError instead of silently counting as zero words. Pure table-rule or horizontal-rule
tokens (runs of - = + | emitted by pandoc for borders) are not counted, as Word counts none."""
import re
import subprocess
import sys

_RULE = re.compile(r"^[-=+|]{3,}$")


def _lines(docx):
    out = subprocess.run(["pandoc", "-f", "docx", "-t", "plain", "--wrap=none", docx],
                         capture_output=True, text=True, check=True).stdout
    return out.splitlines()


def _idx(lines, name, start=0):
    for i in range(start, len(lines)):
        if lines[i].strip().lower() == name:
            return i
    raise ValueError("heading not found in %s: %r" % ("document" if start == 0 else
                                                     "document after line %d" % start, name))


def _words(ls):
    return sum(1 for tok in re.findall(r"\S+", "\n".join(ls)) if not _RULE.match(tok))


END_HEADINGS = ("acknowledgments", "acknowledgements", "author contributions", "statements and declarations")


def _first(lines, name, start):
    for i in range(start, len(lines)):
        if lines[i].strip().lower() == name:
            return i
    return None


def strict_count(docx):
    lines = _lines(docx)
    a = _idx(lines, "introduction")
    r = _idx(lines, "references", a)
    t = _idx(lines, "tables", r)
    ends = [i for i in (_first(lines, h, a) for h in END_HEADINGS) if i is not None and i < r]
    return _words(lines[a:min(ends + [r])]) + _words(lines[t:])


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, strict_count(p))
