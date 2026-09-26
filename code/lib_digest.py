"""Merge every output/revision/*.json into one digest, and compare a digest with the committed one.

The digest also carries numbers the revised text quotes from files that are not revision JSON outputs:
the frozen first-submission digests in output/v1_submitted (Aims 1-2, their statistics, and the eICU-CRD
replication), and a few small result tables. They enter under their own stems, flattened to one level
(nested keys joined with "."), so check_claims can compare them with the text like any other number."""
import csv
import json
from pathlib import Path

# stem: JSON file relative to the output folder (the parent of the revision folder)
FROZEN_JSON = {
    "frozen_aim12": "v1_submitted/results_digest.json",
    "frozen_stats": "v1_submitted/stats_digest.json",
    "frozen_eicu": "v1_submitted/external_eicu_digest.json",
}
# stem: (CSV file relative to the output folder, columns that name a row)
CSV_STEMS = {
    "excluded_vs_included": ("revision/excluded_vs_included.csv", ("comparison", "group")),
    "time_to_first_na": ("revision/time_to_first_non_assessable_by_phenotype.csv", ("phenotype",)),
    "frozen_vent_timecourse": ("v1_submitted/table_no_assessable_timecourse.csv", ("window_h",)),
}
LABEL_FIELDS = ("phenotype", "strategy", "method", "slug", "family", "variable")


def _row_label(item, i):
    parts = [str(item[k]) for k in LABEL_FIELDS if isinstance(item, dict) and k in item]
    return "_".join(parts) if parts else str(i)


def flatten(obj, prefix=""):
    """One-level dict of every scalar in obj. Dict keys are joined with "."; a list of records is keyed by the
    record's label fields (phenotype, strategy, ...) when they are unique within the list, else by position."""
    out = {}
    if isinstance(obj, dict):
        items = [(str(k), v) for k, v in obj.items()]
    elif isinstance(obj, list):
        labels = [_row_label(v, i) for i, v in enumerate(obj)]
        if len(set(labels)) != len(labels):
            labels = [str(i) for i in range(len(obj))]
        items = list(zip(labels, obj))
    else:
        return {prefix: obj}
    for k, v in items:
        out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    return out


def _number(s):
    try:
        f = float(s)
    except ValueError:
        return s
    return int(f) if f.is_integer() and "." not in s else f


def _csv_stem(path, key_cols):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        label = "_".join(r[c] for c in key_cols)
        for col, v in r.items():
            if col not in key_cols and v != "":
                out[f"{label}.{col}"] = _number(v)
    return out


def build_digest(rev_dir):
    rev_dir = Path(rev_dir)
    d = {}
    for f in sorted(rev_dir.glob("*.json")):
        # skip the digest itself and macOS AppleDouble sidecars (._name.json) that the external volume creates
        if f.name != "revision_digest.json" and not f.name.startswith("."):
            d[f.stem] = json.loads(f.read_text(encoding="utf-8"))
    out_dir = rev_dir.parent
    for stem, rel in FROZEN_JSON.items():
        if (out_dir / rel).exists():
            d[stem] = flatten(json.loads((out_dir / rel).read_text(encoding="utf-8")))
    for stem, (rel, key_cols) in CSV_STEMS.items():
        if (out_dir / rel).exists():
            d[stem] = _csv_stem(out_dir / rel, key_cols)
    return d


def digest_text(d):
    return json.dumps(d, indent=1, sort_keys=True, allow_nan=False)


def stale_stems(built, committed):
    """Sorted stems whose content differs between two digests, including stems present in only one."""
    missing = object()
    return sorted(k for k in set(built) | set(committed) if built.get(k, missing) != committed.get(k, missing))
