"""Wording check for the manuscript, supplement and response letter.

    python3 code/tools/lint_language.py [--en-dash] FILE...

Exit code 0 = clean, 1 = at least one hit, 2 = a file could not be read or the usage was wrong.
One line is printed per hit: `file:line: [reason] "matched phrase" ... start of the line`. The matched
phrase is shown because in a long paragraph it is usually beyond the start of the line.

A line is skipped (that line only) when it ends with a suppression marker that carries a reason:
`<!-- lint-ok: reason -->`. A marker in the middle of a line, without a reason (a reason needs at
least one letter or digit), or followed by more text does not suppress anything. Lines whose first non-blank character is `>` (quoted originals in
the response letter) are skipped. The References section is skipped, from its heading to the next
heading of any level. macOS `._*` sidecar files are ignored. After all files have been read, the
number of suppressed lines and of skipped sidecar files is reported on stderr so that both can be
audited.
"""
import os
import re
import sys

# Building blocks for the entries on EEG / pupillometry replacing the examination.
_EEG = r"(?:EEG|pupillometry|neuro-?monitoring|electroencephalograph(?:y|ic))"
_REPLACE = r"replac(?:e|es|ed|ing)"
_SUBSTITUTE = r"(?:substitut(?:e|es|ed|ing)|(?:stand(?:s|ing)?|stood) in)"
_TARGET = r"(?:verbal|GCS|examinations?|exams?|information)"

BANNED = [
    (r"misclassif", "implies a reference standard (R1)"),
    (r"\bgeneral propert", "overgeneralises from one derivation (R2)"),
    (r"\btraps?\b|\btrapp(?:ed|ing)\b", "rhetorical (O5)"),
    (r"\bsilent(ly)?\b", "rhetorical (O5)"),
    (r"\b(went|stayed|goes|going) dark\b|\bdarktimes?\b", "rhetorical (O5)"),
    (r"\bblind spot\b", "rhetorical (O5)"),
    (r"\bperfect\b.{0,20}\b(?:scores?|totals?|GCS)\b", "rhetorical (O5)"),
    (r"\bnormali[sz](?:e|es|ed|ing|ations?)\b.{0,30}\b(?:patients?|scores?)\b", "rhetorical (O5)"),
    (r"\bsickest\b|\bmost[- ]sedated\b", "rhetorical (O5)"),
    (r"\bnear[- ]perfect\b|\bnearly[- ]perfect\b|\bby construction\b", "uninformative calibration claim (R7)"),
    (r"\brecover(y|ed|ing|s)?\b", "verbal-score 'recovery' (R6)"),
    # Cross-validation (5-fold, out-of-fold predictions) is legitimate; the lookbehind exempts only a
    # word-initial "cross" followed by a hyphen, en dash, non-breaking hyphen or space, so that
    # "across validation cohorts" is still flagged. External "validation" is not what eICU-CRD provides.
    (r"(?<!\bcross[-\u2010\u2011\u2013\u00a0 ])\bvalidat(?:e|es|ed|ion|ions|ing)\b",
     "eICU is a replication, not a validation (R5)"),
    # "data-driven" (a compound adjective) is exempt; "driven largely by", "driving", "drove" are not.
    (r"(?<![-\u2010\u2011])\bdriven (?:\w+ ){0,3}by\b|\bdriv(?:es?|ing|ers?)\b|\bdrove\b", "causal wording (R3)"),
    (r"\bnot missing at random\b|\bmissing not at random\b|\bMNAR\b",
     "structural unobservability, not a missingness-at-random claim (R3)"),
    (r"\btrue\b.{0,20}\b(?:GCS|scores?|components?)\b"
     r"|\b(?:actual|real|ground[- ]truth)\b.{0,20}\b(?:GCS|scores?|components?)\b", "no reference standard (R1)"),
    # A plain rule description ("the default replaced the verbal score with 5") is not flagged: without an
    # EEG or pupillometry subject, "verbal" counts only when it names an assessment, response or function.
    # With such a subject before the verb, any verbal, GCS, examination or information target is flagged.
    (r"\breplac(?:e|es|ed|ing)\b.{0,40}\b(?:examinations?|exams?|information"
     r"|verbal (?:assessments?|responses?|responsiveness|functions?))\b"
     r"|\b" + _EEG + r"\b.{0,40}\b" + _REPLACE + r"\b.{0,40}\b" + _TARGET + r"\b",
     "EEG/pupillometry do not replace it (O7)"),
    (r"\b" + _SUBSTITUTE + r"\b.{0,40}\b(?:examinations?|exams?)\b"
     r"|\b" + _EEG + r"\b.{0,40}\b" + _SUBSTITUTE + r"\b.{0,40}\b" + _TARGET + r"\b",
     "EEG/pupillometry do not replace it (O7)"),
    (r"\breplaced by (?:an? |the )?" + _EEG + r"\b", "EEG/pupillometry do not replace it (O7)"),
    (r"\bunassessable\b|\bunexaminable\b|\bnot assessable\b|\bmissing verbal\b", "use 'non-assessable' (O2)"),
    (r"\u2014", "em dash in prose (house style)"),
]
EN_DASH = (r"\u2013", "en dash (response letter house style)")
# The marker only counts at the END of the line and its reason must contain a letter or digit, so
# "<!-- lint-ok: -->", "<!-- lint-ok: - -->" and "<!-- lint-ok: --->" do not suppress anything.
SUPPRESS = re.compile(r"<!--\s*lint-ok:(?=[^>]*?[^\W_])[^>]*?-->\s*$")
USAGE = "usage: lint_language.py [--en-dash] FILE..."


def _phrase(line, m):
    """The matched text, widened to whole words at both ends ("misclassif" -> "misclassifies")."""
    a, b = m.start(), m.end()
    while a > 0 and (line[a - 1].isalnum() or line[a - 1] == "_"):
        a -= 1
    while b < len(line) and (line[b].isalnum() or line[b] == "_"):
        b += 1
    return line[a:b]


def scan(text, en_dash=False, with_match=False):
    """Return (hits, n_suppressed). A hit is (line_no, pattern, reason, line), or with with_match=True
    (line_no, pattern, reason, line, matched_phrase). n_suppressed counts the lines that would have
    produced a hit but end with a valid suppression marker."""
    patterns = BANNED + ([EN_DASH] if en_dash else [])
    hits, suppressed, in_refs = [], 0, False
    for n, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if re.match(r"^#+\s*references\b", s, re.I):
            in_refs = True
            continue
        if in_refs and re.match(r"^#+\s", s):
            in_refs = False
        if in_refs or s.startswith(">"):
            continue
        marker = SUPPRESS.search(line)
        prose = line[:marker.start()] if marker else line  # the reason itself is not linted
        found = []
        for pat, why in patterns:
            m = re.search(pat, prose, re.I)
            if m:
                found.append((pat, why, _phrase(prose, m)))
        if not found:
            continue
        if marker:
            suppressed += 1
            continue
        hits.extend((n, pat, why, s[:120], phrase) if with_match else (n, pat, why, s[:120])
                    for pat, why, phrase in found)
    return hits, suppressed


def lint(text, en_dash=False):
    return scan(text, en_dash=en_dash)[0]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    flags = [a for a in argv if a.startswith("--")]
    files = [a for a in argv if not a.startswith("--")]
    unknown = [a for a in flags if a != "--en-dash"]
    if unknown or not files:
        if unknown:
            print("lint_language: unknown option %s" % " ".join(unknown), file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    en = "--en-dash" in flags
    bad = unreadable = suppressed = sidecars = 0
    for f in files:
        if os.path.basename(f).startswith("._"):  # macOS AppleDouble sidecar, not a document
            sidecars += 1
            continue
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError as e:
            print("lint_language: cannot read %s: %s" % (f, e.strerror or e), file=sys.stderr)
            unreadable += 1
            continue
        hits, n_sup = scan(text, en_dash=en, with_match=True)
        suppressed += n_sup
        for n, _pat, why, s, phrase in hits:
            print(f'{f}:{n}: [{why}] "{phrase}" ... {s}')
            bad += 1
    if suppressed:
        print("%d line(s) suppressed with lint-ok" % suppressed, file=sys.stderr)
    if sidecars:
        print("%d sidecar file(s) skipped" % sidecars, file=sys.stderr)
    return 2 if unreadable else (1 if bad else 0)


if __name__ == "__main__":
    sys.exit(main())
