import importlib.util
import re
import shutil

import pandas as pd
import pytest

from config import PROJ, phenotype_masks

SUPP = PROJ / "manuscript" / "supplement.md"
MS = PROJ / "manuscript" / "manuscript.md"


def _gen():
    spec = importlib.util.spec_from_file_location("supp_tables", PROJ / "code" / "74_build_supplement_tables.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def gen():
    return _gen()


@pytest.fixture(scope="module")
def supp():
    return SUPP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ms():
    return MS.read_text(encoding="utf-8")


def _cited(text, kind="eTable"):
    """Numbers cited as 'eTable N', 'eTables N and M' or 'eTables N, M and K', in order of appearance."""
    out = []
    for m in re.finditer(kind + r"s? (\d+(?:(?:, | and )\d+)*)", text):
        out += [int(k) for k in re.findall(r"\d+", m.group(1))]
    return out


# ---- (a) drift guard -------------------------------------------------------------------------------------
@pytest.mark.requires_study_files("manuscript/supplement.md")
def test_regenerating_gives_byte_identical_blocks(gen, supp, tmp_path):
    copy = tmp_path / "supplement.md"
    shutil.copy(SUPP, copy)
    gen.main(copy)
    regenerated = copy.read_text(encoding="utf-8")
    assert gen.blocks(regenerated) == gen.blocks(supp)
    assert regenerated == supp


@pytest.mark.requires_study_files("manuscript/supplement.md")
def test_every_generator_has_exactly_one_block(gen, supp):
    assert sorted(gen.blocks(supp)) == sorted(gen.TABLES)


@pytest.mark.requires_study_files("manuscript/supplement.md")
def test_render_refuses_a_missing_marker(gen, supp):
    with pytest.raises(ValueError):
        gen.render(supp.replace("<!-- BEGIN eTable 5 -->", ""))


# ---- (b) citations and numbering -------------------------------------------------------------------------
@pytest.mark.requires_study_files("manuscript/supplement.md", "manuscript/manuscript.md")
def test_every_cited_etable_exists_with_a_caption(gen, supp, ms):
    blocks = gen.blocks(supp)
    for n in set(_cited(ms)):
        assert n in blocks, f"eTable {n} is cited in the manuscript but absent from the supplement"
        assert re.search(rf"^\*\*eTable {n}\. .+?\.\*\*", blocks[n].strip(), re.M), f"eTable {n} has no caption"


@pytest.mark.requires_study_files("manuscript/supplement.md", "manuscript/manuscript.md")
def test_every_supplement_etable_is_cited(gen, supp, ms):
    outside = re.sub(r"<!-- BEGIN eTable (\d+) -->.*?<!-- END eTable \1 -->", "", supp, flags=re.S)
    cited = set(_cited(ms)) | set(_cited(outside))
    missing = sorted(set(gen.blocks(supp)) - cited)
    assert not missing, f"eTables never cited: {missing}"


@pytest.mark.requires_study_files("manuscript/supplement.md", "manuscript/manuscript.md")
def test_etables_numbered_in_order_of_first_citation(gen, supp, ms):
    first = list(dict.fromkeys(_cited(ms)))
    assert first == sorted(first), first
    assert sorted(gen.blocks(supp)) == list(range(1, len(gen.blocks(supp)) + 1))


@pytest.mark.requires_study_files("manuscript/supplement.md", "manuscript/manuscript.md")
def test_efigures_cited_in_the_main_text_in_order(supp, ms):
    caps = [int(n) for n in re.findall(r"^\*\*eFigure (\d+)\. ", supp, re.M)]
    assert caps == list(range(1, len(caps) + 1))
    first = list(dict.fromkeys(_cited(ms, "eFigure")))
    assert first == caps, f"main-text first citations {first} do not match eFigures {caps}"


# ---- (c) number format -----------------------------------------------------------------------------------
IDENTIFIER_TABLES = {1, 4, 6}   # codes and item identifiers, not quantities (eTable 4 names an item id)


def _cells(block):
    """(header, cell) for every data cell of every markdown table in a block."""
    out, header = [], None
    for line in block.splitlines():
        if not line.startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if header is None:
            header = cells
        elif not set(line) <= set("|-: "):
            out += list(zip(header, cells))
    return out


@pytest.mark.requires_study_files("manuscript/supplement.md")
def test_thousands_separator_only_from_ten_thousand(gen, supp):
    for n, b in gen.blocks(supp).items():
        for tok in re.findall(r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?![\d.])", b):
            assert int(tok.replace(",", "")) >= 10000, f"eTable {n}: {tok}"
        if n not in IDENTIFIER_TABLES:
            for tok in re.findall(r"(?<![\w.,])\d{5,}(?![\d,])", b):
                pytest.fail(f"eTable {n}: {tok} needs a thousands separator")


# Paired AUROC differences are the one exception to three decimals: when an interval endpoint lies near zero they may
# use four, so that the sign stays visible. Estimate and both endpoints must then share one precision.
DIFF_SENTENCE = re.compile(r"Paired AUROC difference.*?(?:\. (?=[A-Z])|$)", re.M)
TRIPLE = re.compile(r"(-?\d\.\d+) \((-?\d\.\d+) to (-?\d\.\d+)\)")


@pytest.mark.requires_study_files("manuscript/supplement.md")
def test_paired_differences_use_three_or_four_decimals(gen, supp):
    found = 0
    for n, b in gen.blocks(supp).items():
        for sent in DIFF_SENTENCE.findall(b):
            for m in TRIPLE.finditer(sent):
                found += 1
                places = {len(x.split(".")[1]) for x in m.groups()}
                assert len(places) == 1 and places <= {3, 4}, f"eTable {n}: {m.group(0)}"
    assert found >= 3


@pytest.mark.requires_study_files("manuscript/supplement.md")
def test_percent_auroc_and_interval_formats(gen, supp):
    for n, b in gen.blocks(supp).items():
        assert not re.search(r"\(\s*-?\d[\d.]*\s*,\s*-?\d[\d.]*\s*\)", b), f"eTable {n}: interval with a comma"
        assert not re.search(r"\d\s*[–—]\s*\d", b), f"eTable {n}: dash range"
        for head, cell in _cells(b):
            nums = re.findall(r"-?\d+(?:\.\d+)?", cell.replace(",", ""))
            if "AUROC" in head or "QWK" in head:
                for x in nums:
                    assert re.fullmatch(r"-?\d\.\d{3}", x), f"eTable {n} {head}: {cell}"
            elif re.search(r"%\)?$", head) and "(%" not in head and "n (%" not in head:
                for x in nums:
                    assert re.fullmatch(r"-?\d+\.\d", x), f"eTable {n} {head}: {cell}"
            elif re.search(r"n \(%", head):
                pct = re.findall(r"\((-?\d+\.\d+)", cell)
                assert pct and all(re.fullmatch(r"-?\d+\.\d", x) for x in pct), f"eTable {n} {head}: {cell}"


# ---- content rules ---------------------------------------------------------------------------------------
@pytest.mark.requires_study_files("manuscript/supplement.md", "output/revision/aim4.json")
def test_seed_and_width_statements_match_the_code_and_files(supp):
    import json
    from lib_audit import BOOT_SEED
    from config import REV
    assert f"fixed seed ({BOOT_SEED})" in supp
    a = json.loads((REV / "aim4.json").read_text())
    stay, exam = a["learned_qwk_hi"] - a["learned_qwk_lo"], a["learned_qwk_exam_hi"] - a["learned_qwk_exam_lo"]
    assert f"about four times narrower (width {exam:.3f} vs {stay:.3f})" in supp
    assert round(stay / exam) == 4


def test_generator_reads_no_git_ignored_file():
    src = (PROJ / "code" / "74_build_supplement_tables.py").read_text(encoding="utf-8")
    assert "read_parquet" not in src and "INT /" not in src


@pytest.mark.requires_study_files("manuscript/supplement.md", "manuscript/title.txt")
def test_title_matches_the_manuscript(supp):
    title = (PROJ / "manuscript" / "title.txt").read_text(encoding="utf-8").strip()
    assert supp.splitlines()[2] == "## " + title


@pytest.mark.requires_study_files("manuscript/supplement.md")
def test_no_submitted_population_or_revision_history(supp):
    for w in ("previously", "corrected", "original submission", "earlier analysis", "legacy", "12,404",
              "submitted", "patient-level", "upper bound"):
        assert w not in supp.lower(), w


@pytest.mark.requires_study_files("manuscript/supplement.md", "manuscript/manuscript.md")
def test_one_brennan_auroc_interval_everywhere(supp, ms):
    for text in (supp, ms):
        for m in re.finditer(r"0\.820 \(([^)]*)\)", text):
            assert m.group(1) == "0.812 to 0.828", m.group(0)
    assert "0.812 to 0.828" in supp


def test_etable1_codes_match_the_phenotype_definitions(gen):
    key = {"Subarachnoid hemorrhage": "SAH", "Intracerebral hemorrhage": "ICH", "Subdural hemorrhage": "SDH",
           "Acute ischemic stroke": "AIS", "Traumatic brain injury": "TBI", "Anoxic (hypoxic-ischemic) injury": "anoxic"}
    for name, icd9, icd10 in gen.PHENOTYPE_CODES:
        codes = []
        for part in icd9.split(", "):
            if " to " in part:
                a, b = part.split(" to ")
                codes += [(str(k), 9) for k in range(int(a), int(b) + 1)]
            else:
                codes.append((part.replace(".", ""), 9))
        codes += [(c.replace(".", ""), 10) for c in icd10.split(", ")]
        df = pd.DataFrame(codes, columns=["icd_code", "icd_version"])
        hit = phenotype_masks(df)[key[name]]
        assert hit.all(), f"{name}: {df[~hit].icd_code.tolist()} not matched by config.phenotype_masks"


# ---- eTable 18: adjusted phenotype odds ratios -------------------------------------------------------------
def _digest_or():
    import json
    d = json.loads((PROJ / "output" / "v1_submitted" / "stats_digest.json").read_text(encoding="utf-8"))
    return {r["term"]: r for r in d["aim1_adj_or"]}


@pytest.mark.requires_study_files("manuscript/supplement.md", "output/v1_submitted/stats_digest.json")
def test_etable18_reports_each_adjusted_odds_ratio_from_the_digest(gen, supp):
    b = gen.blocks(supp)[18]
    assert b.strip().startswith("**eTable 18. Adjusted odds ratios of ever having a non-assessable verbal examination, "
                                "by phenotype.**")
    assert "adjusted for age and sex" in b and "acute ischemic stroke is the reference" in b
    assert "n = 14,230 stays" in b
    rows = [c for c in _cells(b)]
    assert [h for h, _ in rows[:2]] == ["Phenotype (reference: acute ischemic stroke)", "Adjusted odds ratio (95% CI)"]
    want = _digest_or()
    assert len(rows) == 2 * len(want) == 10
    for (_, lab), (head, cell) in zip(rows[::2], rows[1::2]):
        r = want[("anoxic" if lab == "Anoxic" else lab) + " vs AIS"]
        assert head == "Adjusted odds ratio (95% CI)"
        assert cell == f"{r['OR']:.2f} ({r['lo']:.2f} to {r['hi']:.2f})", (lab, cell)


@pytest.mark.requires_study_files("manuscript/supplement.md", "output/v1_submitted/stats_digest.json")
def test_etable18_has_one_row_per_non_reference_phenotype(gen, supp):
    labs = [c for h, c in _cells(gen.blocks(supp)[18]) if h.startswith("Phenotype")]
    assert labs == ["SAH", "ICH", "SDH", "TBI", "Anoxic"]


@pytest.mark.requires_study_files("manuscript/supplement.md", "output/revision/claims.tsv",
                                  "output/revision/revision_digest.json")
def test_etable18_claims_point_at_the_row_they_register(supp):
    """Digest keys are positional (aim1_adj_or.N.*): each registered literal must sit in the row of the same term."""
    import json
    from config import REV
    dig = json.loads((REV / "revision_digest.json").read_text(encoding="utf-8"))["frozen_stats"]
    seen = set()
    for line in (REV / "claims.tsv").read_text(encoding="utf-8").splitlines():
        f, key, kind, lit = line.split("\t")
        m = re.fullmatch(r"frozen_stats:aim1_adj_or\.(\d)\.(OR|lo|hi)", key)
        if not m or f != "manuscript/supplement.md":
            continue
        term = dig[f"aim1_adj_or.{m.group(1)}.term"]
        lab = "Anoxic" if term.startswith("anoxic") else term.split(" vs ")[0]
        assert lit.startswith(f"| {lab} | "), (key, lit)
        assert kind == "dec2"
        seen.add((m.group(1), m.group(2)))
    assert seen == {(str(i), k) for i in range(5) for k in ("OR", "lo", "hi")}
