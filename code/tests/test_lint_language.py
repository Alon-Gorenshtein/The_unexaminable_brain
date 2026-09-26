import os
import pathlib
import re
import subprocess
import sys

import pytest

import lint_language
from lint_language import BANNED, EN_DASH, lint, scan

TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "lint_language.py"

R1_MISCLASS = "implies a reference standard (R1)"
R2_GENERAL = "overgeneralises from one derivation (R2)"
O5 = "rhetorical (O5)"
R7 = "uninformative calibration claim (R7)"
R6 = "verbal-score 'recovery' (R6)"
R5 = "eICU is a replication, not a validation (R5)"
R3 = "causal wording (R3)"
R3_MAR = "structural unobservability, not a missingness-at-random claim (R3)"
R1_TRUE = "no reference standard (R1)"
O7 = "EEG/pupillometry do not replace it (O7)"
O2 = "use 'non-assessable' (O2)"
EM = "em dash in prose (house style)"


# ---- basic behaviour ---------------------------------------------------------------------

def test_flags_rhetorical_and_causal_terms():
    for s in ["The default misclassifies severity.", "a general property of ICU scoring", "selection trap",
              "the examination went dark", "verbal-score recovery", "an external validation cohort",
              "driven by mechanical ventilation", "near-perfect calibration by construction",
              "EEG replaces the lost examination", "true GCS", "unassessable"]:
        assert lint(s), s


def test_clean_text_and_suppression_pass():
    assert not lint("The default rule assigned a total of 15 to non-assessable examinations.")
    assert not lint("Replication in eICU-CRD used the APACHE components.")
    assert not lint("They said misclassifies <!-- lint-ok: reviewer quotation -->")
    assert not lint("> Original: The term misclassification implies a reference standard.")
    assert not lint("## References\n1. Validation of a practical method for estimating GCS.")


def test_case_insensitive():
    assert lint("MISCLASSIFICATION was common.")
    assert lint("Selection Trap")
    assert lint("The mechanism is mnar.")


# ---- cross-validation is legitimate, external validation is not ----------------------------

def test_cross_validation_is_clean_but_external_validation_is_flagged():
    for s in ["out-of-fold predictions from 5-fold cross-validation",
              "Predictions were cross-validated within each hospital.",
              "The model was cross-validating on 5 folds.",
              "Cross-Validation folds were stratified by hospital.",
              "5-fold cross validation was used.",
              "Nested cross-validations were repeated.",
              "5-fold cross\u2011validation was used.",
              "5-fold cross\u2010validation was used.",
              "5-fold cross\u2013validation was used.",
              "5-fold cross\u00a0validation was used.",
              "5-fold crossvalidation was used."]:
        assert not lint(s), s


def test_lookbehind_exempts_only_a_word_initial_cross():
    for s in ["Across validation cohorts the calibration slope was stable.",
              "The rule was checked across validation sites.",
              "an across-validation comparison",
              "supercross-validation",
              "an external validation cohort", "The model was validated in eICU.",
              "validated in eICU", "Validation of the model in a second database.",
              "The score validates the rule.", "Two validations were run.",
              "cross-validated predictions were validated in eICU", "a cross-hospital validation cohort",
              "validating the model"]:
        hits = lint(s)
        assert hits, s
        assert all(h[2] == R5 for h in hits), (s, hits)


# ---- one entry per family: positive samples and the entry each must trip ----------------------
# Samples are keyed by (reason, position among the BANNED entries that share that reason), so they do
# not depend on the regex text. Each entry needs at least one sample per top-level alternative.

SAMPLES = {
    (R1_MISCLASS, 0): ["The default rule misclassifies severity.", "Misclassification was common."],
    (R2_GENERAL, 0): ["This is a general property of ICU scoring.", "General properties of the derivation."],
    (O5, 0): ["This is a selection trap.", "Two traps in the derivation.",
              "Ventilated patients were trapped at 15.", "The trapping effect was large."],
    (O5, 1): ["The rule silently assigns 15.", "The rule is silent about sedation."],
    (O5, 2): ["The examination went dark.", "The record stayed dark for hours.",
              "The chart goes dark after intubation.", "The examination is going dark.",
              "During darktime the chart was empty."],
    (O5, 3): ["This is a blind spot in the record."],
    (O5, 4): ["The default gives a perfect score.", "It yields a perfect verbal score.",
              "It yields a perfect total score.", "It yields a perfect GCS.",
              "Perfect neurological scores were assigned.", "Perfect totals were assigned."],
    (O5, 5): ["The default normalizes the patient.", "The default normalises the score.",
              "Normalization of the patients was applied.", "Normalisations of scores were applied."],
    (O5, 6): ["The sickest patients were affected.", "The most-sedated patients were excluded.",
              "The most sedated patients were excluded."],
    (R7, 0): ["Calibration was near-perfect.", "Calibration was nearly perfect.",
              "Near perfect calibration was seen.", "Calibration holds by construction."],
    (R6, 0): ["verbal-score recovery was learned.", "The model recovered the verbal component.",
              "It recovers the label.", "recovering the label", "Recover the missing values."],
    (R5, 0): ["The model was validated in eICU.", "an external validation cohort",
              "Two validations were run.", "The rule validates the label.", "validating the model",
              "Validate the rule."],
    (R3, 0): ["The gap was driven by ventilation.", "The gap was driven largely by ventilation.",
              "The gap was driven mainly by sedation.", "The gap was driven almost entirely by sedation.",
              "Sedation drives the gap.", "Ventilation drove the gap.", "The driving factor was sedation.",
              "Sedation is a driver of the gap.", "The main drivers were sedation and intubation."],
    (R3_MAR, 0): ["The values were not missing at random.", "The values were missing not at random.",
                  "The mechanism is MNAR."],
    (R1_TRUE, 0): ["The true GCS is unknown.", "The true scores are unknown.", "The actual score is unknown.",
                   "The real component is unknown.", "The ground truth GCS is unknown.",
                   "No ground-truth score is available."],
    # replace: without a subject (verbal must name an assessment, response or function), then with one
    (O7, 0): ["EEG replaces the lost examination.", "Pupillometry replaced the missing information.",
              "Replacing the verbal response with EEG was tested.", "EEG replaces verbal responsiveness.",
              "EEG replaced the exam.", "Replacing the verbal responses with EEG was tested.",
              "Replacing the verbal assessments was proposed.", "It replaces verbal functions.",
              "EEG replaces the verbal score.", "Pupillometry replaces the verbal component.",
              "Neuromonitoring replaced the GCS.", "Neuro-monitoring replacing the verbal score was proposed.",
              "Electroencephalography could replace the verbal score.",
              "EEG-derived features, if cheap, could replace the verbal score."],
    # substitute or stand in: examination targets, then the subject-anchored form
    (O7, 1): ["EEG can substitute for the examination.", "A substituted exam was used.",
              "Pupillometry stands in for the exam.", "EEG stood in for the neurological examination.",
              "The monitor is standing in for the examination.",
              "EEG could substitute for the verbal examination.", "EEG substitutes for the verbal score.",
              "Pupillometry stood in for the verbal component.",
              "Neuromonitoring is standing in for the verbal score.", "Pupillometry stands in for the GCS."],
    (O7, 2): ["The verbal score was replaced by EEG.", "The value was replaced by pupillometry.",
              "The verbal component can be replaced by an EEG-derived index.",
              "The score was replaced by the neuromonitoring signal."],
    (O2, 0): ["The verbal score was not assessable.", "The score is unassessable.",
              "The component is unexaminable.", "Missing verbal scores were imputed."],
    (EM, 0): ["The default rule \u2014 as applied \u2014 assigns 15."],
}


def _keyed():
    """BANNED entries keyed by (reason, position among the entries sharing that reason)."""
    seen, out = {}, {}
    for pat, why in BANNED:
        k = seen.get(why, 0)
        seen[why] = k + 1
        out[(why, k)] = pat
    return out


def _alternatives(pat):
    """Split a regex on its top-level '|' (outside groups, classes and escapes)."""
    out, cur, depth, in_cls, i = [], [], 0, False, 0
    while i < len(pat):
        c = pat[i]
        if c == "\\":
            cur.append(pat[i:i + 2])
            i += 2
            continue
        if in_cls:
            in_cls = c != "]"
        elif c == "[":
            in_cls = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "|" and depth == 0:
            out.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    out.append("".join(cur))
    return out


def test_alternatives_helper():
    assert _alternatives(r"(?<!a|b)x(?:c|d)|y[|]z\|w") == [r"(?<!a|b)x(?:c|d)", r"y[|]z\|w"]
    assert _alternatives(r"abc") == ["abc"]


def test_every_banned_entry_has_samples():
    assert set(SAMPLES) == set(_keyed()), "add positive samples for each new entry (keyed by reason, position)"


def test_every_top_level_alternative_of_every_entry_has_a_sample():
    for key, pat in _keyed().items():
        for alt in _alternatives(pat):
            assert any(re.search(alt, s, re.I) for s in SAMPLES.get(key, [])), (key, alt)


_CASES = [(key, s) for key, ss in sorted(SAMPLES.items()) for s in ss]


@pytest.mark.parametrize("key,sample", _CASES, ids=["%s#%d %s" % (k[0][:18], k[1], s[:28]) for k, s in _CASES])
def test_each_sample_trips_exactly_its_own_entry(key, sample):
    pat = _keyed()[key]
    hits = lint(sample)
    assert hits, sample
    assert {h[1] for h in hits} == {pat}, hits
    assert {h[2] for h in hits} == {key[0]}, hits


def test_eeg_subject_branches_cover_every_subject_verb_and_target():
    subjects = ["EEG", "Pupillometry", "Neuromonitoring", "Neuro-monitoring", "Electroencephalography",
                "EEG-derived features"]
    verbs = ["replaces", "replaced", "replacing", "could replace", "would substitute for",
             "substituted for", "substituting for", "stands in for", "is standing in for", "stood in for"]
    targets = ["the verbal score", "the verbal component", "the verbal examination", "the GCS",
               "the examination", "the lost information"]
    for subj in subjects:
        for verb in verbs:
            for tgt in targets:
                s = "%s %s %s." % (subj, verb, tgt)
                hits = lint(s)
                assert hits and {h[2] for h in hits} == {O7}, s


def test_eeg_subject_window_is_about_one_clause():
    near = "EEG " + "ab " * 12 + "replaced the verbal score."   # 37 characters between subject and verb
    far = "EEG " + "ab " * 16 + "replaced the verbal score."    # 49 characters: a different clause
    assert lint(near) and not lint(far)
    near = "EEG replaced " + "ab " * 10 + "the verbal score."    # 35 characters between verb and target
    far = "EEG replaced " + "ab " * 16 + "the verbal score."     # 53 characters
    assert lint(near) and not lint(far)


def test_eeg_subject_must_come_before_the_verb_and_a_target_must_follow():
    for s in ["The verbal score was recorded, and EEG was replaced.",
              "EEG replaced the electrodes.",
              "Pupillometry stood in the corridor.",
              "EEG stands in the record."]:
        assert lint(s) == [], s


# ---- near-miss clean lines: neutral wording must not trip -------------------------------------

NEAR_MISS_CLEAN = [
    # reference standard and over-generalisation
    "The rule classified each examination as assessable or non-assessable.",
    "A general description of the ICU score is given in Methods.",
    "The true-positive rate was 0.80.",
    "A value of True indicates a non-assessable verbal component.",
    "The actual number of patients was 500.",
    "Real-time monitoring of the pupil was not available.",
    "Ground truth labels were absent.",
    # rhetorical
    "The score discriminated between the two groups.",
    "The trapezius squeeze was recorded by the nurse.",
    "The trapezoid rule was used to compute the area.",
    "The room fell to silence during the examination.",
    "The pupil was dark-adapted before measurement.",
    "Values were normalized to the cohort median.",
    "Values were normalised to the cohort median.",
    "The model showed perfect separation in the training folds.",
    "A perfect match between rows was required.",
    "Patients with the highest severity of illness were analysed separately.",
    "The deepest sedation category was analysed separately.",
    # calibration
    "Calibration slope was 0.97 (95% CI 0.91 to 1.03).",
    "The construction of the cohort is described in Methods.",
    "The two files were nearly identical.",
    # recovery
    "Missing values were not recoverable from the chart.",
    # validation
    "5-fold cross-validation was used to tune the model.",
    "Records that were invalid were removed.",
    "Input validity checks were applied.",
    "Replication in eICU-CRD used the APACHE components.",
    # causal
    "Ventilation was associated with non-assessable verbal scores.",
    "The association persisted after adjustment.",
    "Values were missing for 12% of patients.",
    "Values were missing at random in the simulation.",
    "The data-driven approach was not used.",
    "The data-driven approach followed by imputation was used.",
    # replacement and terminology
    "Missing values were replaced with the cohort median.",
    "Missing values were replaced by the cohort median.",
    "The default replaced the verbal score with 5.",
    "EEG was recorded, and missing values were replaced by the cohort median.",
    "The EEG was reviewed by a neurologist after the examination.",
    "Pupillometry stands for infrared pupil measurement.",
    "EEG was available in 12% of patients and the verbal score is described in Methods.",
    "Substitute values were taken from the nursing flowsheet.",
    "The stand was placed in the corridor.",
    "The component was non-assessable.",
    "Assessable examinations were retained.",
    # punctuation in normal mode: hyphen and en dash are fine
    "Scores ranged from 3-15 and from 3 \u2013 15 in the second table.",
]


@pytest.mark.parametrize("line", NEAR_MISS_CLEAN)
def test_near_miss_neutral_wording_is_clean(line):
    assert lint(line) == [], lint(line)


def test_expected_positives_that_look_like_near_misses():
    # documented behaviour: these DO trip, so nobody lists them as clean
    for s in ["The GCS recovered after extubation.", "Values recovered from the chart.",
              "The examination was not assessable.", "The score is unexaminable.",
              "Missing verbal scores were imputed.", "It was silent.",
              "The driving licence field was not used.",
              "The default replaced the missing verbal examination with 5.",
              "The default replaced the verbal responses with 5."]:
        assert lint(s), s


# ---- hit structure ------------------------------------------------------------------------

def test_hit_tuple_shape_line_numbers_and_truncation():
    text = "clean line\n\n   The default misclassifies severity.   \n" + "x " * 100 + "trap"
    hits = lint(text)
    assert [h[0] for h in hits] == [3, 4]
    n, pat, why, line = hits[0]
    assert (pat, why) in BANNED
    assert why == R1_MISCLASS
    assert line == "The default misclassifies severity."
    assert len(hits[1][3]) <= 120
    assert all(len(h) == 4 for h in hits)


def test_scan_with_match_adds_the_matched_phrase_as_a_fifth_element():
    hits, _ = scan("The default misclassifies severity, a selection trap.", with_match=True)
    assert all(len(h) == 5 for h in hits)
    assert {(h[2], h[4]) for h in hits} == {(R1_MISCLASS, "misclassifies"), (O5, "trap")}
    # the first four elements are what lint() returns
    assert [h[:4] for h in hits] == scan("The default misclassifies severity, a selection trap.")[0]


def test_matched_phrase_is_the_span_the_pattern_matched_widened_to_whole_words():
    def phrase(s, reason):
        return [h[4] for h in scan(s, with_match=True)[0] if h[2] == reason]
    assert phrase("It gave a perfect verbal score to all.", O5) == ["perfect verbal score"]
    assert phrase("Misclassification was common.", R1_MISCLASS) == ["Misclassification"]
    assert phrase("The sickest and most-sedated patients.", O5) == ["sickest"]
    assert phrase("Calibration was near-perfect throughout.", R7) == ["near-perfect"]
    assert phrase("The gap was driven largely by sedation.", R3) == ["driven largely by"]
    assert phrase("a\u2014b", EM) == ["a\u2014b"]


def test_multiple_patterns_on_one_line_give_multiple_hits():
    hits = lint("A selection trap driven by ventilation.")
    assert {h[2] for h in hits} == {O5, R3}
    assert len({h[1] for h in hits}) == 2
    assert all(h[0] == 1 for h in hits)


def test_empty_and_blank_text():
    assert lint("") == []
    assert lint("\n\n   \n") == []
    assert scan("") == ([], 0)


# ---- suppression marker: anchored at the end of the line, with a reason ----------------------------

def test_suppression_comment_skips_only_its_own_line():
    text = ("first: a selection trap\n"
            "second: a selection trap <!-- lint-ok: reviewer quotation -->\n"
            "third: a selection trap\n")
    assert [h[0] for h in lint(text)] == [1, 3]


def test_suppression_needs_a_lint_ok_marker_with_a_colon():
    assert lint("a selection trap <!-- note: fix later -->")
    assert lint("a selection trap <!-- lint-ok -->")
    assert not lint("a selection trap <!--   lint-ok:   quoted from the reviewer   -->")


def test_suppression_does_not_open_a_skip_region():
    text = "a <!-- lint-ok: x -->\nmisclassifies\nmisclassifies"
    assert [h[0] for h in lint(text)] == [2, 3]


def test_marker_in_the_middle_of_a_line_does_not_suppress():
    s = "A selection trap <!-- lint-ok: quote --> and then it was driven by ventilation and misclassifies."
    hits = lint(s)
    assert {h[2] for h in hits} == {O5, R3, R1_MISCLASS}
    assert lint("<!-- lint-ok: quote --> a selection trap")
    assert scan(s)[1] == 0


def test_marker_followed_by_any_text_does_not_suppress():
    assert lint("a selection trap <!-- lint-ok: quote --> .")
    assert lint("a selection trap <!-- lint-ok: quote --> <b>")
    assert lint("a selection trap <!-- lint-ok: a --> misclassifies <!-- lint-ok: b --> x")


def test_marker_with_an_empty_reason_does_not_suppress():
    for s in ["a selection trap <!-- lint-ok: -->", "a selection trap <!--lint-ok:-->",
              "a selection trap <!-- lint-ok:    -->", "a selection trap <!-- lint-ok:\t-->",
              "a selection trap <!-- lint-ok:--> ", "a selection trap <!-- lint-ok: - -->",
              "a selection trap <!-- lint-ok: --->", "a selection trap <!-- lint-ok: ... -->",
              "a selection trap <!-- lint-ok: __ -->", "a selection trap <!-- lint-ok: -- -->"]:
        assert lint(s), s
        assert scan(s)[1] == 0, s


def test_marker_reason_needs_a_letter_or_digit_but_may_contain_punctuation():
    for s in ["a selection trap <!-- lint-ok: reviewer quotation -->",
              "a selection trap <!-- lint-ok: x -->", "a selection trap <!-- lint-ok: 3 -->",
              "a selection trap <!-- lint-ok: - quoted source -->",
              "a selection trap <!--lint-ok:quoted-->",
              "a selection trap <!-- lint-ok: caf\u00e9 -->"]:
        assert lint(s) == [], s
        assert scan(s) == ([], 1), s


def test_marker_reason_cannot_contain_a_closing_bracket():
    assert lint("a selection trap <!-- lint-ok: a > b -->")


def test_marker_at_the_end_tolerates_trailing_whitespace_and_suppresses_every_hit_on_its_line():
    s = "a selection trap driven by misclassifies <!-- lint-ok: quoted source -->   \t"
    assert lint(s) == []
    assert scan(s) == ([], 1)


def test_marker_reason_is_not_itself_linted():
    s = "The wording is neutral <!-- lint-ok: quoted from a selection trap paper -->"
    assert scan(s) == ([], 0)


def test_scan_counts_only_lines_the_marker_actually_suppressed():
    text = ("a selection trap <!-- lint-ok: quoted source -->\n"           # 1 suppressed
            "clean line <!-- lint-ok: stale marker -->\n"                  # 2 nothing to suppress
            "> a selection trap <!-- lint-ok: quoted source -->\n"         # 3 skipped as a quote
            "misclassifies <!-- lint-ok: -->\n"                            # 4 hit (empty reason)
            "a trap <!-- lint-ok: x --> and misclassifies\n"               # 5 hit (marker not last)
            "## References\n"
            "1. a trap <!-- lint-ok: reference title -->\n"                # 7 skipped as a reference
            "## Tables\n"
            "driven by <!-- lint-ok: quoted source -->\n")                 # 9 suppressed
    hits, n = scan(text)
    assert n == 2
    assert {h[0] for h in hits} == {4, 5}


def test_lint_is_scan_without_the_count():
    text = "a trap\nb <!-- lint-ok: r -->\n"
    assert lint(text) == scan(text)[0]


# ---- a quoted line is skipped only when the > starts the line ------------------------------------

def test_quoted_line_is_skipped_only_when_it_starts_the_line():
    assert not lint("> a selection trap")
    assert not lint(">a selection trap")
    assert not lint("   > a selection trap")
    assert lint("The threshold was > 5 in a selection trap.")
    assert lint("a > b: a selection trap")
    assert lint("a selection trap >")


def test_quoted_block_skips_each_quoted_line_but_not_the_reply():
    text = ("> Original: it misclassifies.\n"
            "> The term implies a reference standard.\n"
            "\n"
            "Response: we no longer use that term, and it misclassifies nothing.\n")
    assert [h[0] for h in lint(text)] == [4]


# ---- the References section is skipped until the next heading of any level -------------------------

def test_references_skipped_until_next_heading_of_any_level():
    for nxt in ["# Tables", "## Tables", "### Tables", "#### Tables", "###### Tables"]:
        text = ("## References\n"
                "1. Validation of a practical method.\n"
                "2. A selection trap.\n"
                "\n"
                "3. Recovery of verbal scores.\n"
                + nxt + "\n"
                "A selection trap in the table.\n")
        assert [h[0] for h in lint(text)] == [7], (nxt, lint(text))


def test_heading_that_ends_references_is_itself_linted():
    text = "## References\n1. A trap.\n## A selection trap\nbody\n"
    assert [h[0] for h in lint(text)] == [3]


def test_references_heading_variants_and_depth():
    for head in ["# References", "### References", "## references", "##References", "## REFERENCES"]:
        assert not lint(head + "\n1. A trap.\n2. Validation of a method.\n"), head


def test_references_only_matches_a_references_heading():
    # text that merely mentions references, or a heading that starts with another word, is linted
    assert lint("The references are a selection trap.")
    assert lint("## Reference standard trap")
    assert lint("## Cross-references\nA selection trap.")


def test_text_before_references_is_linted_and_references_can_repeat():
    text = ("A trap.\n## References\n1. A trap.\n## Methods\nA trap.\n"
            "## References\n2. A trap.\n## Tables\nA trap.\n")
    assert [h[0] for h in lint(text)] == [1, 5, 9]


# ---- the en dash is checked only in --en-dash mode --------------------------------------------------

def test_en_dash_mode_flags_en_dash_and_normal_mode_does_not():
    s = "Scores of 3\u201315 were assigned."
    assert not lint(s)
    assert not lint(s, en_dash=False)
    hits = lint(s, en_dash=True)
    assert [h[1] for h in hits] == [EN_DASH[0]]
    assert hits[0][2] == EN_DASH[1]


def test_en_dash_mode_keeps_all_normal_checks_and_em_dash():
    assert lint("a selection trap", en_dash=True)
    assert lint("a \u2014 b", en_dash=False)
    assert lint("a \u2014 b", en_dash=True)


def test_en_dash_mode_still_skips_quotes_suppressed_lines_and_references():
    text = ("> quoted 3\u201315\n"
            "kept 3\u201315 <!-- lint-ok: range from the source table -->\n"
            "## References\n"
            "1. Pages 3\u201315.\n")
    assert lint(text, en_dash=True) == []


def test_en_dash_pattern_is_not_in_the_default_list():
    assert EN_DASH not in BANNED


# ---- command line --------------------------------------------------------------------------------------

def _run(args, cwd=None, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run([sys.executable, str(TOOL)] + [str(a) for a in args],
                          capture_output=True, text=True, cwd=cwd, env=e)


def test_cli_exit_codes_and_output_format(tmp_path):
    clean = tmp_path / "clean.md"
    clean.write_text("The default rule assigned 15 to non-assessable examinations.\n", encoding="utf-8")
    dirty = tmp_path / "dirty.md"
    dirty.write_text("ok\nThe default misclassifies severity.\n", encoding="utf-8")

    r = _run([clean])
    assert (r.returncode, r.stdout, r.stderr) == (0, "", "")

    r = _run([clean, dirty])
    assert r.returncode == 1
    assert r.stdout.splitlines() == [
        f'{dirty}:2: [implies a reference standard (R1)] "misclassifies" ... The default misclassifies severity.']
    assert "Traceback" not in r.stderr


def test_cli_names_the_matched_phrase_when_it_is_far_into_a_long_paragraph(tmp_path):
    lead = "Non-assessable examinations were recorded for each ventilated patient in the cohort. " * 6
    text = lead + "The default therefore created a selection trap for the sickest patients. " + lead
    assert text.index("trap") > 300
    f = tmp_path / "long.md"
    f.write_text(text + "\n", encoding="utf-8")
    r = _run([f])
    lines = r.stdout.splitlines()
    assert r.returncode == 1 and len(lines) == 2, lines
    trap = [ln for ln in lines if '"trap"' in ln]
    assert len(trap) == 1
    trap = trap[0]
    assert trap.startswith(f'{f}:1: [rhetorical (O5)] "trap" ... Non-assessable examinations')
    assert "selection trap" not in trap  # the start of the line is cut at 120 characters
    assert any('"sickest"' in ln for ln in lines)


def test_cli_quotes_a_multiword_span_and_uses_one_line_per_hit(tmp_path):
    f = tmp_path / "s.md"
    f.write_text("Ok.\n" + "x " * 80 + "the default gives a perfect verbal score. It recovers it.\n", encoding="utf-8")
    r = _run([f])
    lines = r.stdout.splitlines()
    assert r.returncode == 1 and len(lines) == 2, lines
    assert any('"perfect verbal score" ... x x x' in ln for ln in lines)
    assert any('"recovers" ... x x x' in ln for ln in lines)
    assert all(ln.startswith(f"{f}:2: [") for ln in lines)


def test_cli_en_dash_flag(tmp_path):
    f = tmp_path / "r.md"
    f.write_text("Scores of 3\u201315.\n", encoding="utf-8")
    assert _run([f]).returncode == 0
    r = _run([f, "--en-dash"])
    assert r.returncode == 1 and "en dash" in r.stdout
    assert _run(["--en-dash", f]).returncode == 1


def test_cli_missing_file_is_one_line_and_exit_2(tmp_path):
    missing = tmp_path / "nope.md"
    r = _run([missing])
    assert r.returncode == 2
    assert "Traceback" not in r.stderr and "Traceback" not in r.stdout
    msgs = [ln for ln in (r.stderr + r.stdout).splitlines() if ln.strip()]
    assert len(msgs) == 1 and str(missing) in msgs[0], msgs


def test_cli_missing_file_does_not_hide_hits_in_other_files(tmp_path):
    dirty = tmp_path / "dirty.md"
    dirty.write_text("a selection trap\n", encoding="utf-8")
    r = _run([tmp_path / "nope.md", dirty])
    assert r.returncode == 2
    assert "dirty.md:1:" in r.stdout


def test_cli_directory_argument_is_exit_2_not_traceback(tmp_path):
    r = _run([tmp_path])
    assert r.returncode == 2 and "Traceback" not in r.stderr


def test_cli_skips_macos_sidecar_files_and_says_how_many(tmp_path):
    good = tmp_path / "good.md"
    good.write_text("Clean sentence.\n", encoding="utf-8")
    sidecar = tmp_path / "._good.md"
    sidecar.write_bytes(b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X        \x00\x02\xff\xfe misclassifies trap")
    sub = tmp_path / "sub"
    sub.mkdir()
    nested = sub / "._nested.md"
    nested.write_bytes(b"\xff\xfe\x00 misclassifies")
    ghost = tmp_path / "._does_not_exist.md"  # skipped before opening, so absence is not an error

    r = _run([good, sidecar, nested, ghost])
    assert (r.returncode, r.stdout) == (0, "")
    assert r.stderr.splitlines() == ["3 sidecar file(s) skipped"]

    r = _run([good])
    assert (r.returncode, r.stdout, r.stderr) == (0, "", "")


def test_cli_only_skips_basenames_starting_with_dot_underscore(tmp_path):
    d = tmp_path / "._dir"          # a directory called ._dir is not a sidecar FILE
    d.mkdir()
    f = d / "real.md"
    f.write_text("a selection trap\n", encoding="utf-8")
    r = _run([f])
    assert r.returncode == 1 and "sidecar" not in r.stderr
    g = tmp_path / "x._y.md"
    g.write_text("a selection trap\n", encoding="utf-8")
    assert _run([g]).returncode == 1


def test_cli_reports_suppressed_lines_on_stderr_after_all_files(tmp_path):
    a = tmp_path / "a.md"
    a.write_text("a selection trap <!-- lint-ok: quoted source -->\nfine\n", encoding="utf-8")
    b = tmp_path / "b.md"
    b.write_text("driven by <!-- lint-ok: quoted source -->\n"
                 "clean <!-- lint-ok: stale marker -->\n"
                 "misclassifies <!-- lint-ok: reviewer quotation -->\n", encoding="utf-8")
    r = _run([a, b])
    assert (r.returncode, r.stdout) == (0, "")
    assert r.stderr.splitlines() == ["3 line(s) suppressed with lint-ok"]


def test_cli_suppressed_count_does_not_hide_hits_or_change_exit_codes(tmp_path):
    a = tmp_path / "a.md"
    a.write_text("a selection trap <!-- lint-ok: quoted source -->\n"
                 "a selection trap <!-- lint-ok: -->\n", encoding="utf-8")
    r = _run([a])
    assert r.returncode == 1
    assert r.stdout.splitlines() == [f'{a}:2: [rhetorical (O5)] "trap" ... a selection trap <!-- lint-ok: -->']
    assert r.stderr.splitlines() == ["1 line(s) suppressed with lint-ok"]


def test_cli_prints_suppressed_summary_before_sidecar_summary_and_after_read_errors(tmp_path):
    a = tmp_path / "a.md"
    a.write_text("a selection trap <!-- lint-ok: quoted source -->\n", encoding="utf-8")
    side = tmp_path / "._a.md"
    side.write_bytes(b"x")
    missing = tmp_path / "nope.md"
    r = _run([side, missing, a])
    assert r.returncode == 2
    lines = r.stderr.splitlines()
    assert len(lines) == 3 and str(missing) in lines[0], lines
    assert lines[1:] == ["1 line(s) suppressed with lint-ok", "1 sidecar file(s) skipped"]


def test_cli_prints_no_summary_when_nothing_was_suppressed_or_skipped(tmp_path):
    a = tmp_path / "a.md"
    a.write_text("clean <!-- lint-ok: stale marker -->\n", encoding="utf-8")
    r = _run([a])
    assert (r.returncode, r.stdout, r.stderr) == (0, "", "")


def test_cli_reads_utf8_regardless_of_locale_and_survives_bad_bytes(tmp_path):
    f = tmp_path / "u.md"
    f.write_bytes("caf\u00e9 \u2014 a selection trap\n".encode("utf-8"))
    ascii_env = {"LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0",
                 "PYTHONIOENCODING": "utf-8"}
    r = _run([f], env=ascii_env)
    assert r.returncode == 1 and "Traceback" not in r.stderr
    assert "em dash" in r.stdout and "rhetorical" in r.stdout

    bad = tmp_path / "bad.md"
    bad.write_bytes(b"caf\xe9 \xff\xfe a selection trap\n")
    r = _run([bad], env=ascii_env)
    assert r.returncode == 1 and "Traceback" not in r.stderr and "rhetorical" in r.stdout


def test_cli_usage_and_unknown_flag_exit_2(tmp_path):
    f = tmp_path / "f.md"
    f.write_text("Clean.\n", encoding="utf-8")
    r = _run([])
    assert r.returncode == 2 and "usage" in r.stderr.lower()
    r = _run([f, "--endash"])
    assert r.returncode == 2 and "--endash" in r.stderr and "Traceback" not in r.stderr


def test_main_is_importable_and_returns_the_exit_code(tmp_path, capsys):
    f = tmp_path / "f.md"
    f.write_text("a selection trap\n", encoding="utf-8")
    assert lint_language.main([str(f)]) == 1
    assert "rhetorical (O5)" in capsys.readouterr().out
    f.write_text("Clean.\n", encoding="utf-8")
    assert lint_language.main([str(f)]) == 0
