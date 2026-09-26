"""Mark the text that changed between the submitted and the revised manuscript.

    python3 code/tools/mark_changes.py OLD.md NEW.md OUT.md

Every added or changed unit of NEW.md is wrapped as `[unit]{.mark}` (pandoc renders the class as a
highlight, and highlight_to_color.py turns the highlight into colored text); unchanged text and the
markdown around it are left byte for byte as they were, so removing the markup gives NEW.md back.
Deleted text produces nothing.

Units: a sentence inside a paragraph or list item, the text of a heading, a table cell, and a whole
entry of the reference list. Units are compared on normalised text (whitespace collapsed, emphasis
markers and citation superscripts removed, typographic quotes made straight), so a citation that was
only renumbered is not a change.

Sentences and headings are aligned with difflib.SequenceMatcher over the whole document; a unit the
alignment leaves unmatched is still treated as unchanged when an identical, otherwise unused unit
exists in the old text (a moved sentence). A table cell is unchanged when the old text has a cell with
the same row label, column header and value, so a changed number marks only its own cell.
"""
import difflib
import re
import sys
from collections import Counter
from pathlib import Path

# Words that end with a period but do not end a sentence (compared case-insensitively).
ABBREVIATIONS = ("vs", "e.g", "i.e", "et al", "fig", "figs", "cf", "approx", "dr", "mr", "ms", "st",
                 "jr", "no", "vol", "ref", "refs", "resp", "eq", "sect", "ca")
_CITE = re.compile(r"\^[0-9][0-9,\-– ]*\^")
_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
_LIST = re.compile(r"^(\s*(?:\d+[.)]|[-*+])\s+)(.*)$")
_HEADING = re.compile(r"^(#{1,6}\s+)(.*?)(\s*#*\s*)$")
_HR = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
OPEN, CLOSE = "[", "]{.mark}"


def normalise(text):
    """Text used to decide whether two units are the same."""
    t = _CITE.sub("", text.translate(_QUOTES))
    t = t.replace("**", "").replace("*", "").replace("`", "")
    t = re.sub(r"(?<![A-Za-z0-9])_|_(?![A-Za-z0-9])", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _balanced(s):
    """True when emphasis markers, brackets and code spans in `s` are all closed."""
    if s.count("`") % 2 or s.count("[") != s.count("]"):
        return False
    return s.count("**") % 2 == 0 and s.replace("**", "").count("*") % 2 == 0


def _is_abbreviation(before):
    """`before` is the text up to and including a period; True if the period closes an abbreviation."""
    m = re.search(r"([A-Za-z][A-Za-z.]*(?: al)?)\.$", before)
    if not m:
        return False
    word = m.group(1).lower()
    return any(word == a or word.endswith(" " + a) for a in ABBREVIATIONS)


def sentence_spans(text):
    """(start, end) of each sentence in `text`, covering every non-space character exactly once."""
    spans, start, i, n = [], 0, 0, len(text)
    while i < n:
        if text[i] in ".?!":
            j = i + 1
            while True:  # closers that belong to the sentence: ) " ' * and citation superscripts
                if j < n and text[j] in ")\"'”’*]":
                    j += 1
                    continue
                m = _CITE.match(text, j)
                if m:
                    j = m.end()
                    continue
                break
            k = j
            while k < n and text[k] in " \t\n":
                k += 1
            ok = (k > j and k < n and (text[k].isupper() or text[k].isdigit() or text[k] in "*\"“([")
                  and not (text[i] == "." and _is_abbreviation(text[:i + 1]))
                  and not (text[i] == "." and i > 0 and text[i - 1].isdigit() and i + 1 < n and text[i + 1].isdigit())
                  and _balanced(text[start:j]))
            if ok:
                spans.append((start, j))
                start, i = k, k
                continue
            i = j
            continue
        i += 1
    if text[start:].strip():
        end = len(text.rstrip())
        spans.append((start, end))
    return spans


def _split_cells(line):
    """(start, end) of each cell's content in a pipe-table row, excluding the pipes and padding."""
    bars = [m.start() for m in re.finditer(r"(?<!\\)\|", line)]
    cells = []
    for a, b in zip(bars, bars[1:]):
        seg = line[a + 1:b]
        lead = len(seg) - len(seg.lstrip())
        body = seg.strip()
        cells.append((a + 1 + lead, a + 1 + lead + len(body)))
    return cells


def parse(md):
    """Units of a markdown document.

    Returns a list of dicts: kind ('text', 'cell', 'ref'), key (comparison key), and pos (line index,
    start, end) giving where the unit sits in the document's lines."""
    lines = md.split("\n")
    units, section, i = [], "", 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or _HR.match(line) or line.lstrip().startswith(("<!--", "![")):
            i += 1
            continue
        h = _HEADING.match(line)
        if h:
            section = normalise(h.group(2)).lower()
            s, e = h.start(2), h.end(2)
            if line[s:e].strip():
                units.append({"kind": "text", "key": normalise(line[s:e]), "pos": (i, s, e)})
            i += 1
            continue
        if line.lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(i)
                i += 1
            header = [normalise(lines[block[0]][a:b]) for a, b in _split_cells(lines[block[0]])]
            for r in block:
                if _TABLE_SEP.match(lines[r]):
                    continue
                cells = _split_cells(lines[r])
                label = normalise(lines[r][cells[0][0]:cells[0][1]]) if cells else ""
                for c, (a, b) in enumerate(cells):
                    if a == b:
                        continue
                    value = normalise(lines[r][a:b])
                    if r == block[0]:
                        key = ("header", value)
                    elif c == 0:
                        key = ("label", header[0] if header else "", value)
                    else:
                        key = ("cell", label, header[c] if c < len(header) else "", value)
                    units.append({"kind": "cell", "key": key, "pos": (r, a, b)})
            continue
        # paragraph or list: a run of non-blank lines that are not a heading or a table
        block = []
        while (i < len(lines) and lines[i].strip() and not _HEADING.match(lines[i])
               and not lines[i].lstrip().startswith("|") and not _HR.match(lines[i])):
            block.append(i)
            i += 1
        items, cur = [], None  # each item: list of (line, start) pieces, joined with "\n"
        for r in block:
            m = _LIST.match(lines[r])
            if m:
                cur = [(r, len(m.group(1)))]
                items.append(cur)
            elif cur is None:
                cur = [(r, len(lines[r]) - len(lines[r].lstrip()))]
                items.append(cur)
            else:
                cur.append((r, 0))
        for item in items:
            text = "\n".join(lines[r][s:] for r, s in item)
            offsets, acc = [], 0  # map text offsets back to (line, column)
            for r, s in item:
                offsets.append((acc, r, s))
                acc += len(lines[r]) - s + 1

            def locate(off):
                for base, r, s in reversed(offsets):
                    if off >= base:
                        return r, s + off - base
            spans = [(0, len(text.rstrip()))] if section == "references" else sentence_spans(text)
            for a, b in spans:
                (ra, ca), (rb, cb) = locate(a), locate(b)
                kind = "ref" if section == "references" else "text"
                units.append({"kind": kind, "key": normalise(text[a:b]), "pos": (ra, ca, rb, cb)})
    return units


def changed_units(old_md, new_md):
    """Units of `new_md` that are added or changed relative to `old_md`."""
    old, new = parse(old_md), parse(new_md)
    old_cells = {u["key"] for u in old if u["kind"] == "cell"}
    old_text = [u["key"] for u in old if u["kind"] != "cell"]
    new_text = [u for u in new if u["kind"] != "cell"]
    sm = difflib.SequenceMatcher(None, old_text, [u["key"] for u in new_text], autojunk=False)
    matched_new, matched_old = set(), Counter()
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            matched_new.add(blk.b + k)
            matched_old[old_text[blk.a + k]] += 1
    spare = Counter(old_text) - matched_old  # old units the alignment did not use: candidates for moves
    marked = []
    for idx, u in enumerate(new_text):
        if idx in matched_new or not u["key"]:
            continue
        if spare[u["key"]] > 0:
            spare[u["key"]] -= 1
            continue
        marked.append(u)
    marked += [u for u in new if u["kind"] == "cell" and u["key"] not in old_cells]
    return marked


def mark(old_md, new_md):
    """Return (marked markdown, number of spans)."""
    lines = new_md.split("\n")
    inserts = {}  # line -> list of (column, text)
    units = changed_units(old_md, new_md)
    for u in units:
        p = u["pos"]
        if len(p) == 3:
            r, a, b = p
            inserts.setdefault(r, []).extend([(a, OPEN), (b, CLOSE)])
        else:
            ra, ca, rb, cb = p
            inserts.setdefault(ra, []).append((ca, OPEN))
            inserts.setdefault(rb, []).append((cb, CLOSE))
    for r, ins in inserts.items():
        line = lines[r]
        # at one column a closing mark goes before an opening one
        for col, text in sorted(ins, key=lambda t: (t[0], t[1] == OPEN), reverse=True):
            line = line[:col] + text + line[col:]
        lines[r] = line
    return "\n".join(lines), len(units)


def strip_marks(md):
    """Remove every `[...]{.mark}` wrapper, keeping its content."""
    out, stack = list(md), []
    i = 0
    drop = set()
    while i < len(md):
        c = md[i]
        if c == "\\":
            i += 2
            continue
        if c == "[":
            stack.append(i)
        elif c == "]" and stack:
            s = stack.pop()
            if md.startswith(CLOSE, i):
                drop.add(s)
                drop.update(range(i, i + len(CLOSE)))
                i += len(CLOSE)
                continue
        i += 1
    return "".join(ch for k, ch in enumerate(out) if k not in drop)


def main(argv):
    if len(argv) != 4:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    old, new = Path(argv[1]).read_text(encoding="utf-8"), Path(argv[2]).read_text(encoding="utf-8")
    marked, n = mark(old, new)
    if strip_marks(marked) != new:
        print("mark_changes: markup does not round-trip; nothing written", file=sys.stderr)
        return 1
    Path(argv[3]).write_text(marked, encoding="utf-8")
    print(f"{argv[3]}: {n} marked units")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
