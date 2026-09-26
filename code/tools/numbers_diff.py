"""List numeric tokens that appear only in the submitted text or only in the revised text, so
every vanished number is either explained in the response or a mistake.

Run from the repository root. Usage: python3 code/tools/numbers_diff.py <path-tracked-at-jicm-v1>"""
import re
import subprocess
import sys

# Signed numbers and leading-dot decimals are kept whole. A comma followed by exactly three digits is a
# thousands separator ("12,404"); a comma followed by fewer is a list separator ("1,2,3"). A sign counts
# only when it is not glued to a preceding digit or word, so ranges such as "3-15" stay unsigned.
TOK = re.compile(r"(?<![\w.])[-+−]?(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?%?"
                 r"|(?<![\w.])[-+−]?\.\d+%?")


def tokens(text):
    return {t.replace("−", "-") for t in TOK.findall(text)}


def main(argv):
    if len(argv) != 1:
        print("usage: python3 code/tools/numbers_diff.py <path>   (run from the repository root)", file=sys.stderr)
        return 2
    path = argv[0]
    shown = subprocess.run(["git", "show", f"jicm-v1:{path}"], capture_output=True, text=True)
    if shown.returncode != 0:
        print(f"git show jicm-v1:{path} failed: {shown.stderr.strip()}", file=sys.stderr)
        return 1
    try:
        with open(path, encoding="utf-8") as fh:
            new = fh.read()
    except OSError as e:
        print(f"cannot read {path} in the working tree: {e.strerror or e}", file=sys.stderr)
        return 1
    a, b = tokens(shown.stdout), tokens(new)
    print("only in submitted:", sorted(a - b))
    print("only in revised:  ", sorted(b - a))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
