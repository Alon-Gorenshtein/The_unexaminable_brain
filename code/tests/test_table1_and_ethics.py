import csv
import io
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from lib_cohort import hospital_los_days

ROOT = Path(__file__).resolve().parents[2]
MS = ROOT / "manuscript"
BIDMC_PROTOCOL = "2001P001699"
EICU_CERTIFICATION = "1031219-2"
ALLOWED_IDENTIFIERS = {BIDMC_PROTOCOL, EICU_CERTIFICATION}
# Any token of five or more digits (protocol, certification or approval numbers), with letters and a "-n" tail.
IDENTIFIER_LIKE = re.compile(r"[A-Za-z0-9]*\d{5,}[A-Za-z0-9]*(?:-\d+)?")


def _norm_label(s):
    return re.sub(r"[,\s]+", " ", s.replace(", n (%)", "").replace(" n (%)", "")).strip().lower()


def _norm_value(s):
    return s.replace(",", "").strip()


def table1_problems(md, csv_text):
    """Rows of Table 1 in the manuscript that differ from the table the script wrote (label and every cell,
    ignoring thousands separators and the ', n (%)' label suffix), plus rows present on one side only."""
    lines = md.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("**Table 1."))
    block = []
    for l in lines[start + 1:]:
        if l.startswith("|"):
            block.append([c.strip() for c in l.strip().strip("|").split("|")])
        elif block:
            break
    header, body = block[0], block[2:]
    ms = {_norm_label(r[0]): [_norm_value(c) for c in r[1:]] for r in body}
    src = list(csv.reader(io.StringIO(csv_text)))
    want = {_norm_label(r[0]): [_norm_value(c) for c in r[1:]] for r in src[1:]}
    if [h.lower() for h in header[1:]] != [h.lower() for h in src[0][1:]]:
        return ["columns differ"]
    return [k for k in sorted(set(ms) | set(want)) if ms.get(k) != want.get(k)]


def _note(md):
    return next((l for l in md.splitlines() if l.startswith("**Table 1 note.**")), "")


def none_claim_problems(md, digest):
    """The characteristics the Table 1 note says have no missing value must be zero in the digest."""
    m = re.search(r"none for ([^;]+);", _note(md))
    if not m:
        return ["no 'none for ...' clause"]
    keys = {"age": "age", "sex": "sex", "mechanical ventilation": "ventilation", "hospital mortality": "mortality",
            "hospital length of stay": "hospital_los"}
    names = re.split(r",\s*|\s+or\s+", m.group(1))
    bad = [f"unknown characteristic {n!r}" for n in names if n.strip() not in keys]
    for n in names:
        k = keys.get(n.strip())
        for prefix in ("overall", "comparator"):
            if k and digest[f"{prefix}_{k}"] != 0:
                bad.append(f"{prefix}_{k} is {digest[f'{prefix}_{k}']}, not none")
    return bad


@pytest.mark.requires_study_files("manuscript/manuscript.md", "output/tables/table1.csv")
def test_table1_in_the_manuscript_matches_the_table_the_script_wrote():
    md = (MS / "manuscript.md").read_text(encoding="utf-8")
    text = (ROOT / "output" / "tables" / "table1.csv").read_text(encoding="utf-8")
    assert table1_problems(md, text) == []
    assert "hospital los days median (iqr)" in [_norm_label(r[0]) for r in csv.reader(io.StringIO(text))]


def test_table1_check_fails_on_a_changed_cell_a_missing_row_and_a_new_column():
    md = "**Table 1. T.**\n\n| Characteristic | A | B |\n|---|---|---|\n| N | 1,000 | 2 |\n| Female | 5 (50.0) | 1 (50.0) |\n\nNext\n"
    ok = ",A,B\nN,1000,2\n\"Female, n (%)\",5 (50.0),1 (50.0)\n"
    assert table1_problems(md, ok) == []
    assert table1_problems(md, ok.replace("5 (50.0)", "6 (50.0)")) == ["female"]
    assert table1_problems(md, ok + "Age,1,2\n") == ["age"]
    assert table1_problems(md, ok.replace(",A,B", ",A,C")) == ["columns differ"]


@pytest.mark.requires_study_files("manuscript/manuscript.md", "output/revision/table1_missing.json")
def test_the_characteristics_the_note_calls_complete_have_no_missing_values():
    md = (MS / "manuscript.md").read_text(encoding="utf-8")
    digest = json.loads((ROOT / "output" / "revision" / "table1_missing.json").read_text(encoding="utf-8"))
    assert none_claim_problems(md, digest) == []


def test_none_claim_check_fails_when_a_listed_characteristic_has_missing_values():
    md = "**Table 1 note.** Missing values: none for age, sex or hospital mortality; insurance, 5 (9).\n"
    zero = {f"{p}_{k}": 0 for p in ("overall", "comparator") for k in ("age", "sex", "mortality")}
    assert none_claim_problems(md, zero) == []
    assert none_claim_problems(md, {**zero, "comparator_sex": 3}) == ["comparator_sex is 3, not none"]
    assert none_claim_problems(md.replace("sex", "height"), zero) == ["unknown characteristic 'height'"]
    assert none_claim_problems("**Table 1. T.**\n", zero) == ["no 'none for ...' clause"]


def ethics_problems(md):
    """The Ethics paragraph of the Methods and the Ethical considerations statement must be one identical
    paragraph that names the BIDMC protocol and the eICU-CRD Safe Harbor certification, claims no approval of
    eICU-CRD, and carries no other identifier-like number (only the two verified identifiers are allowed)."""
    lines = md.splitlines()

    def para(heading):
        i = lines.index(heading)
        return next(l for l in lines[i + 1:] if l.strip())

    a, b = para("### Ethics"), para("### Ethical considerations")
    bad = [] if a == b else ["Ethics and Ethical considerations differ"]
    bad += [f"missing {x}" for x in sorted(ALLOWED_IDENTIFIERS) if x not in a]
    bad += [f"unverified identifier {x}" for x in sorted(set(IDENTIFIER_LIKE.findall(a + " " + b)) - ALLOWED_IDENTIFIERS)]
    if re.search(r"eICU-CRD\s+(was|were)\s+approved", a):
        bad.append("claims eICU-CRD was approved")
    return bad


@pytest.mark.requires_study_files("manuscript/manuscript.md")
def test_ethics_paragraph_is_identical_in_both_places_and_carries_only_the_verified_identifiers():
    assert ethics_problems((MS / "manuscript.md").read_text(encoding="utf-8")) == []


def test_ethics_check_fails_on_a_divergent_copy_a_missing_identifier_an_extra_number_and_an_eicu_approval_claim():
    p = f"MIMIC-IV (BIDMC protocol {BIDMC_PROTOCOL}). eICU-CRD Safe Harbor (certification no. {EICU_CERTIFICATION})."
    md = f"### Ethics\n\n{p}\n\n### Ethical considerations\n\n{p}\n"
    assert ethics_problems(md) == []
    assert ethics_problems(md.replace("### Ethical considerations\n\n" + p, "### Ethical considerations\n\n" + p + " x")) == [
        "Ethics and Ethical considerations differ"]
    assert ethics_problems(md.replace(BIDMC_PROTOCOL, "")) == [f"missing {BIDMC_PROTOCOL}"]
    extra = md.replace("(BIDMC", "(MIT protocol 1234567890; BIDMC")
    assert ethics_problems(extra) == ["unverified identifier 1234567890"]
    assert ethics_problems(md.replace(BIDMC_PROTOCOL, "2001P001698")) == [f"missing {BIDMC_PROTOCOL}", "unverified identifier 2001P001698"]
    assert ethics_problems(md.replace("eICU-CRD Safe", "eICU-CRD was approved. Safe")) == ["claims eICU-CRD was approved"]


def _stays():
    """Three stays admitted 2150-01-01 10:00: a same-day death whose discharge time precedes admission, an
    ordinary stay whose discharge is the latest end time, and a stay without ICU or death times."""
    return pd.DataFrame({
        "admittime": ["2150-01-01 10:00:00"] * 3,
        "dischtime": ["2150-01-01 00:00:00", "2150-01-03 10:00:00", "2150-01-02 10:00:00"],
        "deathtime": ["2150-01-01 15:00:00", None, None],
        "outtime": pd.to_datetime(["2150-01-01 18:00:00", "2150-01-02 10:00:00", None])})


def test_hospital_los_is_positive_when_the_discharge_time_precedes_the_admission_time():
    df = _stays()
    naive = (pd.to_datetime(df["dischtime"]) - pd.to_datetime(df["admittime"])).dt.total_seconds() / 86400
    assert naive.iloc[0] < 0                                   # the recorded discharge alone gives a negative stay
    los = hospital_los_days(df)
    assert los.round(4).tolist() == [round(8 / 24, 4), 2.0, 1.0]   # 18:00 ICU discharge; 3 Jan discharge; 2 Jan discharge
    assert (los >= 0).all()


def test_hospital_los_is_missing_not_zero_when_no_end_time_is_recorded():
    df = _stays().iloc[[2]].assign(dischtime=None)
    assert hospital_los_days(df).isna().all()


@pytest.mark.requires_study_files("output/intermediate/cohort.parquet")
def test_hospital_los_has_no_negative_value_in_the_staged_cohort():
    c = pd.read_parquet(ROOT / "output" / "intermediate" / "cohort.parquet")
    assert hospital_los_days(c).dropna().min() >= 0
