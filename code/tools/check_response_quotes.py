"""Check that every quoted Original and Revised passage in the response letter is verbatim.

    python3 code/tools/check_response_quotes.py [--response PATH] [--ref TAG]

Convention in manuscript/response_to_reviewers.md
-------------------------------------------------
A quotation that must be checked is a bold label line, alone on its line,

    **Original:**      the quotation comes from the submitted manuscript or supplement
    **Revised:**       the quotation comes from the revised manuscript or supplement

followed (blank lines allowed) by the quotation: one or more block quotes (consecutive lines that
start with `>`) and pipe tables (consecutive lines that start with `|`), separated by single blank
lines. The quotation ends at the first line that starts with neither. A block quote may hold several
paragraphs (separated by a bare `>` line); each paragraph is checked on its own.

A quoted table is written as a normal pipe table (header row, separator row, body rows) so that it
renders as a table. It passes when one table of the source files has the same header row and contains
every quoted body row, compared cell by cell after normalisation.

Inside a paragraph, "..." or the ellipsis character separates fragments of one source paragraph. Each
fragment must occur in the source after normalisation:
- Original: `git show TAG:manuscript/manuscript.md` or `git show TAG:manuscript/supplement.md`
  (either file; TAG defaults to jicm-v1, the submitted version);
- Revised: the working-tree manuscript/manuscript.md or manuscript/supplement.md (either file).

Normalisation, applied to the quotation and to the source alike: `^n^` citation superscripts are
removed; markdown emphasis markers (`*`, `_`) and backslash escapes are removed; curly quotes and
apostrophes become straight ones; non-breaking and other Unicode spaces become spaces; en dashes and
minus signs become hyphens; runs of whitespace collapse to one space.

Failures are attributed to the most recent heading. Only a heading of the exact form `### Comment 1.4.`
(any heading level, final period optional) names a comment; any other heading names a section, and prose
lines (bold or not) that mention a comment never change the attribution.

Every label must be followed by a block quote, and at least one quotation must be found; otherwise
the check fails. Each failure is printed with the comment or section it belongs to and a snippet.
Exit code 0 = every fragment found, 1 = at least one failure, 2 = a file could not be read.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABEL = re.compile(r"^\*\*(Original|Revised):\*\*\s*$")
COMMENT_HEADING = re.compile(r"^#{1,6}\s+Comment (\d+\.\d+)\.?\s*$")
HEADING = re.compile(r"^#{1,6}\s+(.*\S)\s*$")
ELLIPSIS = re.compile(r"\.\.\.|…")

_QUOTES = {"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "´": "'"}
_DASHES = {"–": "-", "−": "-", "‐": "-", "‑": "-"}
_SPACES = re.compile(r"[  -​  　]")


def normalize(text):
    """Apply the typography normalisation described in the module docstring."""
    t = re.sub(r"\^[^^\s]*\^", "", text)          # ^12^, ^13-15^, ^1,2^ superscripts
    t = re.sub(r"\\(.)", r"\1", t)                 # markdown backslash escapes
    t = t.replace("*", "").replace("_", "")        # emphasis markers
    for a, b in {**_QUOTES, **_DASHES}.items():
        t = t.replace(a, b)
    t = _SPACES.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def fragments(paragraph):
    """Split one quoted paragraph on ellipses; drop empty or punctuation-only pieces."""
    out = []
    for piece in ELLIPSIS.split(paragraph):
        p = normalize(piece).strip(" ,;:")
        if re.search(r"\w", p):
            out.append(p)
    return out


def _cells(row):
    """Normalised cells of one pipe-table row."""
    return tuple(normalize(c) for c in re.split(r"(?<!\\)\|", row.strip().strip("|")))


_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def source_tables(text):
    """[(header cells, set of body-row cells)] for every pipe table in a markdown source."""
    out, block = [], []
    for line in text.splitlines() + [""]:
        if line.lstrip().startswith("|"):
            block.append(line)
            continue
        if len(block) >= 2 and _SEP.match(block[1]):
            out.append((_cells(block[0]), {_cells(r) for r in block[2:]}))
        block = []
    return out


def extract_quotes(response_text):
    """Return (quotes, problems). A quote is (where, kind, content, line_no); kind is Original or Revised
    and content is a paragraph (str) or a table (list of row strings: header, separator, body rows)."""
    lines = response_text.splitlines()
    quotes, problems = [], []
    where = "(before the first comment)"
    i = 0

    def starts(k, ch):
        return k < len(lines) and lines[k].lstrip().startswith(ch)
    while i < len(lines):
        line = lines[i]
        h = HEADING.match(line.strip())
        c = COMMENT_HEADING.match(line.strip())
        if c:
            where = "Comment " + c.group(1)
        elif h:
            where = h.group(1)
        m = LABEL.match(line.strip())
        if not m:
            i += 1
            continue
        kind, start = m.group(1), i + 1
        j = start
        while j < len(lines) and not lines[j].strip():
            j += 1
        found = False
        while starts(j, ">") or starts(j, "|"):
            found = True
            if starts(j, "|"):
                first, rows = j + 1, []
                while starts(j, "|"):
                    rows.append(lines[j])
                    j += 1
                quotes.append((where, kind, rows, first))
            else:
                block = []
                while starts(j, ">"):
                    block.append((j + 1, re.sub(r"^\s*>\s?", "", lines[j])))
                    j += 1
                para, first = [], None
                for n, content in block + [(None, "")]:
                    if content.strip():
                        para.append(content)
                        first = first or n
                    elif para:
                        quotes.append((where, kind, " ".join(para), first))
                        para, first = [], None
            if j + 1 < len(lines) and not lines[j].strip() and (starts(j + 1, ">") or starts(j + 1, "|")):
                j += 1  # one blank line between parts of the same quotation
        if not found:
            problems.append(f"{where} [{kind}] line {i + 1}: label is not followed by a block quote or table")
        i = j
    return quotes, problems


def check_table(rows, tables):
    """Problems with one quoted table against the source tables (empty list means it is verbatim)."""
    if len(rows) < 3 or not _SEP.match(rows[1]):
        return ["quoted table needs a header row, a separator row and at least one body row"]
    head, body = _cells(rows[0]), [_cells(r) for r in rows[2:]]
    cands = [s for h, s in tables if h == head]
    if not cands:
        return [f"no source table has the header \"{' | '.join(head)}\""]
    best = min(([r for r in body if r not in s] for s in cands), key=len)
    return [f"row not in the source table: \"{' | '.join(r)}\"" for r in best]


def check(response_text, original_sources, revised_sources):
    """Return (n_fragments_checked, failures)."""
    srcs = {"Original": [normalize(s) for s in original_sources],
            "Revised": [normalize(s) for s in revised_sources]}
    tabs = {"Original": [t for s in original_sources for t in source_tables(s)],
            "Revised": [t for s in revised_sources for t in source_tables(s)]}
    quotes, failures = extract_quotes(response_text)
    n = 0
    for where, kind, para, line_no in quotes:
        if isinstance(para, list):
            n += len(para) - 2
            for prob in check_table(para, tabs[kind]):
                failures.append(f"{where} [{kind}] line {line_no}: {prob} (in the {kind.lower()} "
                                f"manuscript or supplement)")
            continue
        frs = fragments(para)
        if not frs:
            failures.append(f"{where} [{kind}] line {line_no}: empty quotation")
        for fr in frs:
            n += 1
            if not any(fr in s for s in srcs[kind]):
                snippet = fr if len(fr) <= 110 else fr[:107] + "..."
                failures.append(f"{where} [{kind}] line {line_no}: not found in the {kind.lower()} "
                                f"manuscript or supplement: \"{snippet}\"")
    if not quotes:
        failures.append("no Original or Revised quotation found (check the label convention)")
    return n, failures


def _git_show(ref, path, root):
    return subprocess.run(["git", "show", f"{ref}:{path}"], cwd=root, capture_output=True,
                          text=True, check=True).stdout


def main(argv=None, root=ROOT):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--response", default="manuscript/response_to_reviewers.md")
    ap.add_argument("--ref", default="jicm-v1", help="git ref of the submitted version")
    a = ap.parse_args(argv)
    try:
        response = (root / a.response).read_text(encoding="utf-8")
        revised = [(root / p).read_text(encoding="utf-8")
                   for p in ("manuscript/manuscript.md", "manuscript/supplement.md")]
        original = [_git_show(a.ref, p, root) for p in ("manuscript/manuscript.md", "manuscript/supplement.md")]
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"check_response_quotes: cannot read a source: {e}", file=sys.stderr)
        return 2
    n, failures = check(response, original, revised)
    if failures:
        print("\n".join(failures))
        print(f"{len(failures)} failure(s) in {n} fragment(s)")
        return 1
    print(f"all {n} quoted fragments found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
