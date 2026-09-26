"""Assert the counts the paper is built on against the regenerated digest.

    python3 code/tools/verify_headline_counts.py

Exits 1 and prints one line per count that moved."""
import json
import sys
from pathlib import Path

REV = Path(__file__).resolve().parents[2] / "output" / "revision"
EXPECTED = {("audit", "sel_n_total"): 14192, ("audit", "sel_deaths_total"): 2702,
            ("audit", "sel_n_excluded"): 5509, ("audit", "sel_deaths_excluded"): 1966,
            ("feasible", "official_n"): 14215, ("cohort_accounting", "neuro_first_stay_n"): 14272,
            ("cohort_accounting", "with_verbal_n"): 14230, ("aim4", "n_exams"): 248738,
            ("aim4", "n_patients"): 12351}


def check(digest):
    return [f"{sec}:{key} = {digest.get(sec, {}).get(key)!r}, expected {want}"
            for (sec, key), want in EXPECTED.items() if digest.get(sec, {}).get(key) != want]


if __name__ == "__main__":
    problems = check(json.loads((REV / "revision_digest.json").read_text()))
    print("\n".join(problems) or "headline counts hold")
    sys.exit(1 if problems else 0)
