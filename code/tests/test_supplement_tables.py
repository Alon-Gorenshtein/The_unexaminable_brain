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


@pytest.mark.requires_study_files("manuscript/supplement.md", "output/revision/excluded_vs_included.csv")
def test_etable3_reports_deaths(gen, supp):
    body = supp.split("<!-- BEGIN eTable 3 -->")[1].split("<!-- END eTable 3 -->")[0]
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in body.splitlines() if l.startswith("|")]
    header, data = rows[0], rows[2:]
    assert header[header.index("Stays, n") + 1] == "Deaths, n"
    e = pd.read_csv(PROJ / "output" / "revision" / "excluded_vs_included.csv")
    assert [r[3] for r in data] == [gen.n_(d) for d in e.deaths]


# ---- definition sensitivity: eTable 1 note, eTable 2 panel C, eTable 12 panel C, eTable 7 label ------------
def _definition_sensitivity():
    import json
    from config import REV
    return json.loads((REV / "definition_sensitivity.json").read_text(encoding="utf-8"))


def _panel(block, start, stop=None):
    """The text of `block` from the line that starts with `start` to the line that starts with `stop` (or the end)."""
    lines = block.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(start))
    j = next((k for k in range(i + 1, len(lines)) if stop and lines[k].startswith(stop)), len(lines))
    return "\n".join(lines[i:j])


def _rows(text):
    """(header, {first cell: remaining cells}) of the first markdown table in `text`."""
    t = [[c.strip() for c in l.strip().strip("|").split("|")] for l in text.splitlines() if l.startswith("|")]
    return t[0], {r[0]: r[1:] for r in t[2:]}, [r[0] for r in t[2:]]


PANEL_C_ETABLE2 = "Panel C. Main estimates with the primary codes and with codes specific to the labeled injury (eTable 1)."


@pytest.mark.requires_study_files("manuscript/supplement.md", "output/revision/definition_sensitivity.json")
def test_panel_c_values_match_definition_sensitivity(gen, supp):
    d = _definition_sensitivity()
    b = gen.blocks(supp)[2]
    header, rows, order = _rows(_panel(b, PANEL_C_ETABLE2))
    assert header == ["Estimate", "Primary codes", "Specific codes"]

    def ever(ph):
        return lambda col: f"{gen.n_(d[f'{col}_ever_{ph}_n'])} / {gen.p1(d[f'{col}_ever_{ph}_pct'])}"

    strat = {"official": "official first-day GCS", "reconstructed": "default-to-15", "impute1": "verbal imputed as 1",
             "gcs_t": "eye-plus-motor sum (T)", "brennan": "Brennan estimate"}
    want = [("Stays with a verbal entry, n / ever non-assessable, %",
             lambda c: f"{gen.n_(d[f'{c}_analytic_n'])} / {gen.p1(d[f'{c}_ever_pct'])}")]
    want += [(f"{lab}, n / ever non-assessable, %", ever(k)) for k, lab in gen.PHENO]
    want += [(f"{lab}, ventilated stays, median % of verbal examinations non-assessable",
              (lambda k: lambda c: gen.p1(d[f"{c}_vent_frac_{k}_pct"]))(k)) for k, lab in gen.PHENO]
    want += [("Anoxic, in-hospital mortality, %", lambda c: gen.p1(d[f"{c}_mortality_anoxic_pct"])),
             ("Anoxic, ventilated, %", lambda c: gen.p1(d[f"{c}_vent_anoxic_pct"])),
             ("Stays with an official first-day GCS, n / total of 15, %",
              lambda c: f"{gen.n_(d[f'{c}_official_n'])} / {gen.p1(d[f'{c}_official_15_pct'])}"),
             ("Official first-day GCS of 15 assigned by the default rule, n / eye-plus-motor sum below 10, %",
              lambda c: f"{gen.n_(d[f'{c}_default_15_n'])} / {gen.p1(d[f'{c}_default_15_impossible_pct'])}"),
             ("Audit stays, n", lambda c: gen.n_(d[f"{c}_audit_n"])),
             ("Deaths among audit stays, n", lambda c: gen.n_(d[f"{c}_audit_deaths"])),
             ("Non-assessable selected examinations, n", lambda c: gen.n_(d[f"{c}_nonassess_n"])),
             ("Deaths excluded by complete-case handling, %", lambda c: gen.p1(d[f"{c}_cc_pct_deaths_excluded"]))]
    want += [(f"AUROC, {lab}", (lambda s: lambda c: gen.a3(d[f"{c}_auroc_{s}"]))(s)) for s, lab in strat.items()]
    want += [("Paired AUROC difference, Brennan estimate minus official first-day GCS (95% CI)",
              lambda c: f"{gen.a3(d[f'{c}_diff_mean'])} ({gen.a3(d[f'{c}_diff_lo'])} to {gen.a3(d[f'{c}_diff_hi'])})"),
             ("Observed-to-expected ratio, non-assessable selected examinations, official first-day GCS",
              lambda c: gen.d2(d[f"{c}_oe_nonassess_official"])),
             ("Observed-to-expected ratio, non-assessable selected examinations, Brennan estimate",
              lambda c: gen.d2(d[f"{c}_oe_nonassess_brennan"]))]
    assert order == [lab for lab, _ in want]
    for lab, f in want:
        assert rows[lab] == [f("primary"), f("specific")], lab
    # the headline figures that the manuscript quotes
    assert rows["Anoxic, n / ever non-assessable, %"] == ["1282 / 66.1", "719 / 91.8"]
    assert rows["Stays with a verbal entry, n / ever non-assessable, %"] == ["14,230 / 45.2", "12,628 / 45.1"]
    # lower case after the comma, except for a name
    for lab in order:
        head, _, tail = lab.partition(", ")
        assert not tail[:1].isupper() or tail.startswith("Brennan"), lab


@pytest.mark.requires_study_files("manuscript/supplement.md", "output/revision/definition_sensitivity.json")
def test_etable2_panel_c_note_carries_the_removed_stays_and_the_comparator(gen, supp):
    d = _definition_sensitivity()
    note = _panel(gen.blocks(supp)[2], PANEL_C_ETABLE2).split("\n\n")[-1]
    assert (f"Of the {gen.n_(d['primary_eligible_n'])} eligible stays, {gen.n_(d['removed_n'])} "
            f"({gen.p1(d['removed_pct'])}%) entered only through non-specific codes "
            f"({gen.n_(d['removed_TBI_n'])} traumatic, {gen.n_(d['removed_AIS_n'])} ischemic, "
            f"{gen.n_(d['removed_anoxic_n'])} anoxic, {gen.n_(d['removed_SDH_n'])} subdural; in-hospital mortality "
            f"{gen.p1(d['removed_mortality_pct'])}%), leaving {gen.n_(d['specific_eligible_n'])}. Of the removed stays, "
            f"{gen.n_(d['removed_icd9_n'])} were coded in ICD-9 and {gen.n_(d['removed_icd10_n'])} in ICD-10 "
            f"({gen.p1(d['removed_icd9_pct'])}% and {gen.p1(d['removed_icd10_pct'])}% of all stays coded in that "
            "version).") in note
    assert "Eligible stays:" not in note and "with the primary codes and" not in note
    assert note.endswith(" AUROC, area under the receiver operating characteristic curve; GCS, Glasgow Coma Scale.")
    assert ("Stays that entered only through non-specific codes were removed from the cohort and not added to the "
            f"comparator; adding them moves the comparator's {gen.p1(d['comparator_ever_pct'])}% to "
            f"{gen.p1(d['comparator_ever_moved_pct'])}%. Other stays were relabeled by the hierarchy applied to their "
            "remaining codes.") in note


@pytest.mark.requires_study_files("manuscript/supplement.md", "output/revision/definition_sensitivity.json")
def test_etable12_panel_c_present(gen, supp):
    d = _definition_sensitivity()
    b = gen.blocks(supp)[12]
    assert b.strip().startswith("**eTable 12. Sensitivity analyses of resampling, cross-validation folds and "
                                "treatment-covariate timing.**")
    cap = "Panel C. Treatment covariates recorded at any time during the ICU stay or by 24 hours after ICU admission"
    text = _panel(b, cap)
    assert text.splitlines()[0] == cap
    header, rows, order = _rows(text)
    assert header == ["Strategy", "AUROC, any time during the stay", "AUROC, first 24 hours",
                      "Difference, first 24 hours minus any time"]
    slugs = ["official", "reconstructed", "impute1", "gcs_t", "brennan"]
    labels = ["Official first-day GCS", "Default-to-15", "Verbal imputed as 1", "Eye-plus-motor sum (T)",
              "Brennan estimate"]
    assert len(order) == 6
    for s, lab in zip(slugs, labels):
        a, f = d[f"primary_auroc_{s}"], d[f"firstday_auroc_{s}"]
        assert rows[lab] == [gen.a3(a), gen.a3(f), gen.d4(f - a)], lab
    last = "AUROC difference, Brennan estimate minus official first-day GCS"
    assert order[-1] == last
    assert rows[last] == [gen.a3(d["primary_diff_mean"]), gen.a3(d["firstday_diff_mean"]),
                          gen.d4(d["firstday_diff_mean"] - d["primary_diff_mean"])]
    note = text.split("\n\n")[-1]
    assert (f"A vasopressor infusion was recorded in {gen.p1(d['stay_vaso_pct'])}% of stays at any time and "
            f"{gen.p1(d['firstday_vaso_pct'])}% in the first 24 hours; mechanical ventilation in "
            f"{gen.p1(d['stay_vent_pct'])}% and {gen.p1(d['firstday_vent_pct'])}%. An exposure counts in the first 24 "
            "hours when it started no later than 24 hours after ICU admission, including one started before "
            "admission.") in note
    assert (f"{gen.a3(d['firstday_diff_mean'])} ({gen.a3(d['firstday_diff_lo'])} to "
            f"{gen.a3(d['firstday_diff_hi'])})") in note
    assert note.endswith("Exposures in the first 24 hours can still follow the selected examination: the earliest "
                         f"exposure started after it in {gen.p1(d['firstday_vaso_after_exam_pct'])}% of stays with a "
                         f"first-day vasopressor infusion and {gen.p1(d['firstday_vent_after_exam_pct'])}% of stays "
                         "with first-day ventilation.")


@pytest.mark.requires_study_files("manuscript/supplement.md")
def test_etable1_note_names_every_nonspecific_code(gen, supp):
    b = gen.blocks(supp)[1]
    note = b.split("\n\n")[-1]
    assert not note.startswith("|")
    for code in ("348.5", "G93.82", "433.x0", "434.x0", "ICD-9 802", "ICD-9 800, 801, 803 and 804 with a fourth digit "
                 "of 0 or 5", "432.0", "432.9", "I62.1", "I62.9", "G93.6", "I65 and I66", "S02", "348.82"):
        assert code in note, code
    for code in ("348.1", "G93.1", "433.x1", "434.x1", "I63", "ICD-9 850 to 854, ICD-9 800, 801, 803 and 804 with any "
                 "other fourth digit, and ICD-10 S06", "432.1", "I62.0"):
        assert code in note, code
    assert "without mention of infarction" in note and "without infarction)" in note
    assert ("denote skull fractures (for 804, fractures of the skull or face with other bones) without mention of "
            "intracranial injury") in note
    assert "The ICD-9 and ICD-10 lists are not exact counterparts:" in note
    assert "eTable 2, panel C repeats the main estimates with the specific codes only" in note


@pytest.mark.requires_study_files("manuscript/supplement.md")
def test_etable2_title_and_etable18_caption(gen, supp):
    assert gen.blocks(supp)[2].strip().startswith(
        "**eTable 2. Diagnostic overlap among the acute brain-injury phenotypes and results under alternative "
        "phenotype-assignment rules and code definitions.**")
    cap = gen.blocks(supp)[18].split("\n\n")[0]
    assert cap.endswith(" Phenotype labels use the primary code definition (eTable 1); eTable 2, panel C gives the "
                        "percentages with specific codes.")


@pytest.mark.requires_study_files("manuscript/supplement.md")
def test_etable7_uses_mean_score_entered(gen, supp):
    b = gen.blocks(supp)[7]
    header = next(l for l in b.splitlines() if l.startswith("| Selection rule | Handling rule"))
    assert "Mean score entered" in header and "Mean GCS" not in b
    assert ("Mean score entered is the mean total the model received; with the eye-plus-motor sum (T), a "
            "non-assessable examination enters as that sum (range, 2 to 10) and an assessable one as its total "
            "(range, 3 to 15).") in b.split("Panel A.")[0]
    assert "Mean GCS" in gen.blocks(supp)[9]      # eTable 9 panel B has no T rule and keeps its label


@pytest.mark.requires_study_files("manuscript/supplement.md")
def test_etable16_note_states_why_overall_calibration_is_close_to_ideal(gen, supp):
    b = gen.blocks(supp)[16]
    assert ("Over all audit stays, the observed-to-expected ratio is close to 1 for every rule, as expected for "
            "parsimonious logistic models fit on large stratified folds; the subgroup rows are the informative "
            "comparison.") in b
    assert "reproduce the overall mortality" not in b


@pytest.mark.requires_study_files("manuscript/supplement.md")
def test_hand_edited_supplement_sentences(supp):
    assert ("**Calibration.** Out-of-fold predictions from a parsimonious logistic model with an intercept, fit on "
            "large folds stratified on in-hospital death, were expected to approximate the overall mortality, so "
            "calibration over all stays was expected to be close to ideal for every rule; it did not distinguish "
            "them.") in supp
    assert ("The analysis code, including the scripts for cohort derivation, statistical analyses, sensitivity "
            "analyses and figure generation, is available at "
            "https://github.com/Alon-Gorenshtein/The_unexaminable_brain/releases/tag/jicm-r2.") in supp
    assert "that reproduces every table and figure" not in supp
    assert "was entered into the model specification in place of the total GCS" in supp
    assert ("any invasive mechanical ventilation during the stay. A sensitivity analysis limited both treatment "
            "covariates to exposures started no later than 24 hours after ICU admission (eTable 12, panel C).") in supp
    s11 = supp.split("## S1.1")[1].split("## S1.2")[0].strip().split("\n\n")[1]
    assert s11.endswith("formed the general-ICU comparator. Some codes are not specific to the labeled injury (notes "
                        "to eTable 1), so the main estimates were repeated with specific codes only (eTable 2, "
                        "panel C).")


def test_readme_lists_77_before_digest():
    text = (PROJ / "code" / "README.md").read_text(encoding="utf-8")
    assert "77_definition_sensitivity.py" in text
    assert text.index("77_definition_sensitivity.py") < text.index("72_revision_digest.py")
    assert text.index("76_mortality_cv_by_patient.py") < text.index("77_definition_sensitivity.py")
