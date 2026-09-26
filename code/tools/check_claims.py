"""Every number that appears in the prose is registered in output/revision/claims.tsv and must
(1) appear in the named file, and (2) equal the value in the digest, formatted the same way.
Numbers are matched as whole tokens, so 7.4% is not found inside 97.4% and 404 is not found inside 12,404,
and a literal with context words ("12 patients") is anchored at its own number edges.

Usage: python3 code/tools/check_claims.py [claims.tsv]"""
import csv
import functools
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from lib_digest import build_digest, stale_stems  # noqa: E402
from lib_fmt import KINDS, fmt  # noqa: E402

NUMBER_START = "0123456789.%-"
NUMBER_END = "0123456789.%"


def _minus(s):
    return s.replace("\u2212", "-")


# Sign rule: a hyphen or U+2212 directly before the digits is a minus sign unless a digit or word character precedes
# it, so "3-15" and "2020-2021" stay ranges; a number without a minus never matches a negative one, and vice versa.
def has_token(text, lit):
    """True if `lit` occurs in `text` anchored at its own number edges: a literal that begins with a digit, '.', '%'
    or '-' is not preceded by a word character, '.', or digit-plus-comma; one that ends with a digit, '.' or '%' is
    not followed by a digit, '.'+digit or ','+digit. Edges that are letters or brackets match plainly."""
    text, lit = _minus(text), _minus(lit.strip())
    if not lit:
        return False
    pat = re.escape(lit)
    if lit[0] in NUMBER_START:
        pat = r"(?<![\w.])(?<!\d,)" + pat
    if lit[-1] in NUMBER_END:
        pat += r"(?!\d)(?!\.\d)(?!,\d)"
    for m in re.finditer(pat, text):
        i = m.start()
        unsigned_number = lit[0] in "0123456789."
        signed_in_text = i >= 1 and text[i - 1] == "-" and (i < 2 or not re.match(r"\w", text[i - 2]))
        if not (unsigned_number and signed_in_text):
            return True
    return False


def check_rows(rows, digest, read_text):
    bad = []
    for f, key, kind, lit in rows:
        if not lit.strip():
            bad.append(f"{key}: empty literal")
            continue
        if kind not in KINDS:
            bad.append(f"{key}: unknown kind {kind!r} (expected one of {', '.join(KINDS)})")
            continue
        stem, sep, k = key.partition(":")
        if not sep or stem not in digest or k not in digest[stem]:
            bad.append(f"{key}: not in digest")
            continue
        try:
            want = fmt(digest[stem][k], kind)
        except (TypeError, ValueError) as e:
            bad.append(f"{key}: cannot format digest value {digest[stem][k]!r} as {kind!r} ({e})")
            continue
        if not has_token(lit, want):
            bad.append(f"{key}: digest says {want}, registered literal is {lit!r}")
        try:
            text = read_text(f)
        except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as e:
            bad.append(f"{key}: cannot read {f} ({e})")
            continue
        if not has_token(text, lit):
            bad.append(f"{key}: {lit!r} not found in {f}")
    return bad


def parse_claims(fh):
    """Read tab-separated `file, key, kind, literal` rows; return (rows, problems). Blank lines are skipped."""
    rows, bad = [], []
    reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
    for r in reader:
        if not any(c.strip() for c in r):
            continue
        if len(r) != 4:
            bad.append(f"claims line {reader.line_num}: expected 4 tab-separated fields "
                       f"(file, key, kind, literal), got {len(r)}")
            continue
        rows.append(tuple(r))
    return rows, bad


def _read(path, root=ROOT):
    p = root / path
    if p.suffix == ".docx":
        if shutil.which("pandoc") is None:
            raise RuntimeError(f"pandoc is required to read {path} and was not found on PATH")
        return subprocess.run(["pandoc", "-f", "docx", "-t", "plain", "--wrap=none", str(p)],
                              capture_output=True, text=True, check=True).stdout.replace("\n", " ")
    return p.read_text(encoding="utf-8").replace("\n", " ")


def main(root=ROOT, claims_path=None):
    rev = root / "output" / "revision"
    digest_path = rev / "revision_digest.json"
    if not digest_path.exists():
        print("digest is missing: run code/72_revision_digest.py")
        return 1
    digest = json.loads(digest_path.read_text(encoding="utf-8"))
    stale = stale_stems(build_digest(rev), digest)
    if stale:
        print("digest is stale: re-run code/72_revision_digest.py")
        print("stems that differ: " + ", ".join(stale))
        return 1
    claims_file = claims_path or rev / "claims.tsv"
    try:
        with open(claims_file, newline="", encoding="utf-8") as fh:
            rows, problems = parse_claims(fh)
    except OSError as e:
        print(f"cannot read claims file {claims_file}: {e.strerror or e}")
        return 1
    problems += check_rows(rows, digest, functools.lru_cache(maxsize=None)(lambda f: _read(f, root)))
    if not rows:
        print("WARNING: 0 claims registered", file=sys.stderr)
    print("\n".join(problems) or f"all {len(rows)} claims hold")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(claims_path=Path(sys.argv[1]) if len(sys.argv) > 1 else None))
