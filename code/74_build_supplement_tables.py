"""Write every supplementary data table (eTable N) into manuscript/supplement.md.

Each table lives between the lines `<!-- BEGIN eTable N -->` and `<!-- END eTable N -->`. Everything between the
markers (caption, panels, notes) is produced here from committed output files; the prose outside the markers is
edited by hand. Running the script replaces only the marker blocks, so it can be rerun at any time.

    python3 code/74_build_supplement_tables.py [path/to/supplement.md]

Number style (AMA): a comma only from 10,000; percentages to one decimal; AUROC and kappa to three decimals;
odds ratios, ratios and slopes to two decimals; intervals written "a to b"."""
import json
import re
import sys

import pandas as pd

from config import (IT, NMB, PROJ, REV, SEDATIVES, TAB, VASOPRESSORS, VENT_PROC,
                    VERBAL_NONASSESS, VERBAL_SCORE)
from lib_audit import BOOT_B
from lib_gcs_strategies import _kramer_add

SUPPLEMENT = PROJ / "manuscript" / "supplement.md"
V1 = PROJ / "output" / "v1_submitted"
BEGIN, END = "<!-- BEGIN eTable {n} -->", "<!-- END eTable {n} -->"

PHENO = [("SAH", "SAH"), ("ICH", "ICH"), ("SDH", "SDH"), ("TBI", "TBI"), ("AIS", "AIS"), ("anoxic", "Anoxic")]
ABBR = ("AIS, acute ischemic stroke; ICH, intracerebral hemorrhage; SAH, subarachnoid hemorrhage; "
        "SDH, subdural hemorrhage; TBI, traumatic brain injury.")
STRATEGY = {"official": "Official first-day GCS", "reconstructed": "Default-to-15", "impute1": "Verbal imputed as 1",
            "gcs_t": "Eye-plus-motor sum (T)", "brennan": "Brennan estimate", "complete_case": "Complete-case"}
BOOT = str(BOOT_B)  # resamples behind every audit interval (lib_audit.BOOT_B)
SLUGS = ("official", "reconstructed", "impute1", "gcs_t", "brennan")


# ---------------------------------------------------------------- formatting
def n_(v):
    n = int(round(float(v)))
    return f"{n:,}" if abs(n) >= 10000 else str(n)


def p1(v):
    return f"{float(v):.1f}"


def _signed(s):
    return s[1:] if re.fullmatch(r"-0\.0+", s) else s


def a3(v):
    return _signed(f"{float(v):.3f}")


def d2(v):
    return _signed(f"{float(v):.2f}")


def d4(v):
    """Paired AUROC differences whose interval endpoints lie near zero: four decimals, so the sign is visible."""
    return _signed(f"{float(v):.4f}")


def ci(lo, hi, f):
    return f"{f(lo)} to {f(hi)}"


def est_ci(v, lo, hi, f):
    return f"{f(v)} ({ci(lo, hi, f)})"


def or_(v):
    """Odds ratios: two decimals, one decimal from 10 upward."""
    return f"{float(v):.1f}" if float(v) >= 10 else d2(v)


def pval(p):
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}"[1:] if p < 0.01 else f"{p:.2f}"[1:]


def table(header, rows, align=None):
    align = align or ["---"] + ["---:"] * (len(header) - 1)
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def jload(name):
    return json.loads((REV / name).read_text(encoding="utf-8"))


def block(caption, *parts):
    return "\n\n".join([caption, *[p for p in parts if p]])


# ---------------------------------------------------------------- eTables
PHENOTYPE_CODES = [("Subarachnoid hemorrhage", "430", "I60"), ("Intracerebral hemorrhage", "431", "I61"),
                   ("Subdural hemorrhage", "432", "I62"), ("Acute ischemic stroke", "433, 434", "I63"),
                   ("Traumatic brain injury", "800 to 804, 850 to 854", "S06"),
                   ("Anoxic (hypoxic-ischemic) injury", "348.1, 348.5", "G93.1, G93.82")]


def t_codes():
    return block("**eTable 1. Diagnosis codes used to define acute brain-injury phenotypes.** Codes are matched as "
                 "prefixes on every diagnosis code recorded for the hospitalization, in any position.",
                 table(["Phenotype", "ICD-9", "ICD-10"], PHENOTYPE_CODES, ["---", "---", "---"]))


def t_overlap():
    ca = jload("cohort_accounting.json")
    pairs = pd.read_csv(REV / "phenotype_overlap_pairs.csv", index_col=0)
    h = pd.read_csv(REV / "hierarchy_sensitivity.csv")
    rules = [("primary", "Primary hierarchy"), ("reversed", "Reversed hierarchy"),
             ("single_only", "Single-phenotype stays"), ("first_listed_dx", "First-listed diagnosis"),
             ("any_code", "Any code")]
    a_rows = [[lab] + [n_(pairs.loc[k, k2]) for k2, _ in PHENO] for k, lab in PHENO]
    b_rows = []
    for k, lab in PHENO:
        row = [lab]
        for r, _ in rules:
            x = h[(h.rule == r) & (h.phenotype == k)].iloc[0]
            row.append(f"{n_(x.n)} / {p1(x.ever_na_pct)}")
        b_rows.append(row)
    tot = {r: int(h[h.rule == r].n.sum()) for r, _ in rules}
    assert tot["primary"] == tot["reversed"] == ca["with_verbal_n"]
    b_rows.append(["Stays labelled"] + [n_(tot[r]) if r != "any_code" else "Not applicable" for r, _ in rules])
    b_rows.append(["Spearman correlation with the primary ordering", "Reference"]
                  + [a3(ca[f"rank_corr_{r}"]) for r, _ in rules[1:]])
    first = tot["first_listed_dx"]
    return block(
        "**eTable 2. Diagnostic overlap among the acute brain-injury phenotypes and results under alternative "
        "phenotype-assignment rules.**",
        f"Panel A. Stays with codes for each pair of phenotypes, before the hierarchy (n = {n_(ca['overlap_total_n'])} "
        "stays). The diagonal gives the stays with any code for that phenotype.",
        table(["Phenotype"] + [lab for _, lab in PHENO], a_rows),
        f"Of the {n_(ca['overlap_total_n'])} stays, {n_(ca['overlap_1_n'])} carried codes for one phenotype, "
        f"{n_(ca['overlap_2_n'])} for two and {n_(ca['overlap_3plus_n'])} for three or more "
        f"({p1(ca['overlap_multi_pct'])}% with two or more).",
        f"Panel B. Stays assigned to each phenotype and the percentage of those stays ever non-assessable, under the "
        f"primary hierarchy and four alternative rules (analytic cohort, n = {n_(ca['with_verbal_n'])} stays). "
        "Cells are stays, n / ever non-assessable, %.",
        table(["Phenotype"] + [lab for _, lab in rules], b_rows),
        "Primary hierarchy: subarachnoid, then intracerebral, subdural, traumatic, ischemic, anoxic. Reversed "
        "hierarchy: the same order reversed. Single-phenotype stays: only stays with codes for one phenotype. "
        "First-listed diagnosis: the phenotype of the first-listed diagnosis code; stays whose first-listed code is "
        f"not an acute brain injury are unlabelled, so this rule labels {n_(first)} of {n_(ca['with_verbal_n'])} "
        f"stays ({p1(100 * first / ca['with_verbal_n'])}%). Any code: a stay contributes to every phenotype for which "
        "it carries a code, so the columns overlap. The Spearman correlation compares the ranking of the six "
        "phenotypes by the percentage ever non-assessable with the ranking under the primary hierarchy. The "
        f"{n_(ca['overlap_1_n'])} single-phenotype stays of panel A (of {n_(ca['overlap_total_n'])}) and the "
        f"{n_(tot['single_only'])} of panel B (of the {n_(ca['with_verbal_n'])} stays with a verbal entry) differ only "
        "in the denominator. " + ABBR)


def t_excluded():
    e = pd.read_csv(REV / "excluded_vs_included.csv").set_index(["comparison", "group"])
    lab = {("verbal_documented", "included"): ("Verbal GCS documentation", "Documented (analytic cohort)"),
           ("verbal_documented", "excluded"): ("Verbal GCS documentation", "Not documented (excluded)"),
           ("audit_subset", "included"): ("Complete first-day examination", "Present (audit subset)"),
           ("audit_subset", "excluded"): ("Complete first-day examination", "Absent (outside the audit subset)")}
    rows = []
    for key, (comp, grp) in lab.items():
        r = e.loc[key]
        rows.append([comp, grp, n_(r.n), f"{r.age_median:.0f} ({r.age_q1:.0f} to {r.age_q3:.0f})", p1(r.female_pct),
                     p1(r.ventilated_pct), p1(r.mortality_pct), p1(r.icu_los_median_days)])
    return block(
        "**eTable 3. Characteristics of stays excluded for no verbal GCS documentation and of stays outside the "
        "audit subset, with the stays retained.** Stays are the first ICU stay of each hospitalization with an acute "
        "brain-injury code. The second comparison is within the analytic cohort. IQR, interquartile range; LOS, "
        "length of stay.",
        table(["Comparison", "Group", "Stays, n", "Age, median (IQR), y", "Female, %", "Mechanically ventilated, %",
               "In-hospital mortality, %", "ICU LOS, median, d"], rows, ["---", "---"] + ["---:"] * 6))


def t_values():
    vc = pd.read_csv(REV / "verbal_value_counts.csv")
    rows = []
    for v, k in zip(vc.value, vc.n):
        if v in VERBAL_NONASSESS:
            rows.append([v, "No (non-assessable)", "Not scored", n_(k)])
        else:
            rows.append([v, "Yes", str(VERBAL_SCORE[v]), n_(k)])
    return block(
        "**eTable 4. Recorded values of the GCS verbal component (item 223900) and their frequency across all "
        "charted entries in MIMIC-IV.** \"No Response-ETT\" records that the verbal response could not be observed "
        "because of an endotracheal or tracheostomy tube; it is not a verbal score of 1. \"No Response\" is an "
        "assessable entry with a verbal score of 1.",
        table(["Verbal value", "Assessable", "Verbal score", "Entries, n"], rows, ["---", "---", "---:", "---:"]))


def t_broadened():
    b = pd.read_csv(TAB / "etable4_broadened_def.csv")
    name = dict(PHENO + [("comparator", "Comparator")])
    rows = [[name[r.phenotype], p1(r.pct_ever_primary), p1(r.pct_ever_broad)] for r in b.itertuples()]
    same = list(b.sort_values("pct_ever_primary").phenotype) == list(b.sort_values("pct_ever_broad").phenotype)
    return block(
        "**eTable 5. Burden of non-assessable verbal examinations under a broadened definition.** Percentage of stays "
        "ever non-assessable under the primary definition (\"No Response-ETT\") and under a broadened definition that "
        "also counts \"No Response\" charted during an active invasive mechanical-ventilation episode. The ordering of "
        f"the phenotypes is {'the same' if same else 'different'} under the two definitions. " + ABBR,
        table(["Phenotype", "Ever non-assessable, primary, %", "Ever non-assessable, broadened, %"], rows))


def _ids(d):
    names = {}
    for k, v in d.items():
        names.setdefault(v.replace("_push", ""), []).append(str(k))
    return ", ".join(f"{' / '.join(ks)} ({nm})" for nm, ks in names.items())


def t_identifiers():
    vent = [k for k, v in VENT_PROC.items() if v == "invasive_vent"]
    rows = [["GCS eye / verbal / motor", "chartevents",
             f"{IT['gcs_eye']} / {IT['gcs_verbal']} / {IT['gcs_motor']}"],
            ["Richmond Agitation-Sedation Scale", "chartevents", str(IT["rass"])],
            ["Invasive mechanical ventilation", "procedureevents", ", ".join(map(str, vent))],
            ["Sedative and analgesic infusions", "inputevents", _ids(SEDATIVES)],
            ["Neuromuscular-blocking infusions", "inputevents", _ids(NMB)],
            ["Vasopressor infusions", "inputevents", _ids(VASOPRESSORS)]]
    return block("**eTable 6. Source item identifiers (MIMIC-IV).**",
                 table(["Variable", "Table", "Item identifier(s)"], rows, ["---", "---", "---"]))


def t_rules():
    a = pd.read_csv(REV / "audit_by_rule.csv")
    fe = jload("feasible.json")
    rules = [("lowest_em", "Lowest eye-plus-motor (primary)"), ("earliest", "Earliest (closest to ICU admission)"),
             ("latest", "Latest in the first 24 hours"), ("carryforward_lowest_em", "Lowest eye-plus-motor, carry-forward^a^")]
    rows = []
    for r, lab in rules:
        for slug in (*SLUGS, "complete_case"):
            x = a[(a.rule == r) & (a.slug == slug)]
            if x.empty:
                continue
            x = x.iloc[0]
            strat = STRATEGY[slug] + (" (different stays)^b^" if slug == "complete_case" else "")
            rows.append([lab, strat, n_(x.n), n_(x.deaths), p1(x.mortality_pct), est_ci(x.auroc, x.lo, x.hi, a3),
                         a3(x.brier), d2(x.mean_gcs)])
    b_rows = [[lab, n_(fe[f"{r}_na_n"]), f"{n_(fe[f'{r}_na_impossible_n'])} ({p1(fe[f'{r}_na_impossible_pct'])})"]
              for r, lab in rules]
    cf = a[a.rule == "carryforward_lowest_em"].set_index("slug")
    pr = a[a.rule == "lowest_em"].set_index("slug")
    same3 = all(a3(cf.loc[s, "auroc"]) == a3(pr.loc[s, "auroc"]) for s in cf.index)
    n_cf = int(cf.loc["brennan", "n"])
    return block(
        "**eTable 7. In-hospital-mortality model and feasible-range counts under alternative examination-selection "
        "rules (audit subset).** Each stay contributes one examination with same-time eye, motor and verbal entries "
        "between ICU admission and 24 hours. The official first-day GCS is a first-day minimum that does not depend "
        "on the examination selected, so it is shown with the primary rule and the carry-forward variant only. AUROC "
        f"is out-of-fold with a 95% CI from {BOOT} stay-level bootstrap resamples.",
        "Panel A. Mortality model",
        table(["Selection rule", "Handling rule", "Stays, n", "Deaths, n", "Mortality, %", "AUROC (95% CI)", "Brier",
               "Mean GCS"], rows, ["---", "---"] + ["---:"] * 6),
        "Panel B. Non-assessable selected examinations whose eye-plus-motor sum was below 10, so that a total of 15 "
        "is impossible",
        table(["Selection rule", "Non-assessable selected examinations, n", "Eye-plus-motor sum below 10, n (%)"],
              b_rows),
        f"^a^Examinations taken from the charting rows of the official derivation, in which absent components are "
        f"carried forward for up to 6 hours; the same {n_(n_cf)} stays as the primary rule. Every AUROC "
        f"{'equals' if same3 else 'differs from'} the primary rule to three decimals. ^b^Complete-case analysis keeps "
        "only stays whose selected examination was assessable, a population with lower mortality; its AUROC is not "
        "comparable with the other rows. AUROC, area under the receiver operating characteristic curve.")


def t_brennan():
    rows, prev = [], None
    for em in range(2, 11):
        add = _kramer_add(em)
        if prev and prev[1] == add:
            prev[0].append(em)
        else:
            prev = ([em], add)
            rows.append(prev)
    body = [[f"{g[0]}" if len(g) == 1 else f"{g[0]} to {g[-1]}", f"+{add}"] for g, add in rows]
    return block(
        "**eTable 8. The Brennan eye-and-motor estimate of the verbal score.** When the verbal component is "
        "non-assessable, the eye-plus-motor sum determines the imputed verbal points added to obtain an estimated "
        "total GCS (Brennan et al, J Neurosurg 2020).",
        table(["Eye-plus-motor sum", "Imputed verbal points"], body))


def t_model():
    a4, cs = jload("aim4.json"), jload("cluster_sensitivity.json")
    m = [("learned", "Sedation-aware model"), ("learned_no_rass", "Sedation-aware model without RASS"),
         ("brennan", "Brennan estimate (implied verbal score)"), ("fixed1", "Fixed verbal score of 1")]
    rows = [[lab, est_ci(a4[f"{k}_qwk"], a4[f"{k}_qwk_lo"], a4[f"{k}_qwk_hi"], a3),
             ci(a4[f"{k}_qwk_exam_lo"], a4[f"{k}_qwk_exam_hi"], a3),
             est_ci(100 * a4[f"{k}_acc"], 100 * a4[f"{k}_acc_lo"], 100 * a4[f"{k}_acc_hi"], p1)] for k, lab in m]
    b_rows = [["Sedation-aware model", n_(jload("audit.json")["brennan_lowest_em_n"]),
               est_ci(a4["learned_downstream_auroc"], a4["learned_downstream_lo"], a4["learned_downstream_hi"], a3),
               d2(a4["learned_downstream_mean_gcs"])],
              ["Brennan estimate", n_(jload("audit.json")["brennan_lowest_em_n"]),
               est_ci(a4["brennan_downstream_auroc"], a4["brennan_downstream_lo"], a4["brennan_downstream_hi"], a3),
               d2(jload("audit.json")["brennan_lowest_em_mean_gcs"])]]
    return block(
        "**eTable 9. Sedation-aware model for the charted verbal score.** The model was trained and evaluated only in "
        "assessable examinations, where a verbal score is charted. It cannot be evaluated in non-assessable "
        "examinations, the examinations to which it would be applied, because no verbal score is charted there.",
        f"Panel A. Agreement with the charted verbal score in assessable examinations of the first 72 hours "
        f"({n_(a4['n_exams'])} examinations from {n_(cs['aim4_n_stays'])} ICU stays of {n_(cs['aim4_n_subjects'])} "
        "patients; out-of-fold predictions from 5-fold cross-validation grouped by patient)",
        table(["Verbal-score estimate", "QWK (95% CI), stay-level", "QWK 95% CI, examination-level",
               "Accuracy, % (95% CI), stay-level"], rows, ["---", "---:", "---:", "---:"]),
        "Folds were assigned by patient, so no patient had examinations in both the training and the test fold. "
        f"Difference in QWK, sedation-aware model minus Brennan estimate: stay-level 95% CI "
        f"{ci(a4['qwk_learned_minus_brennan_lo'], a4['qwk_learned_minus_brennan_hi'], a3)}. The stay-level interval "
        "resamples ICU stays with all of their examinations. The examination-level interval is shown for comparison; it "
        "ignores the correlation of examinations within a stay and is therefore narrower.",
        "Panel B. In-hospital-mortality model in the audit subset when the non-assessable verbal component of the "
        "selected examination is assigned the model's prediction or the Brennan estimate",
        table(["Verbal-score estimate", "Stays, n", "AUROC (95% CI)", "Mean GCS"], b_rows),
        f"Paired AUROC difference, model minus Brennan estimate: "
        f"{est_ci(a4['learned_minus_brennan_auroc_mean'], a4['learned_minus_brennan_auroc_lo'], a4['learned_minus_brennan_auroc_hi'], d4)}. "
        f"AUROC intervals are from {BOOT} stay-level bootstrap resamples. QWK, quadratic weighted kappa; RASS, Richmond "
        "Agitation-Sedation Scale.")


def t_motor():
    t, s = pd.read_csv(REV / "motor_only.csv"), jload("motor_only.json")
    na_b = t[(t.scope == "nonassessable") & (t.slug == "brennan")].iloc[0]
    c = pd.read_csv(REV / "comparability.csv")
    cmp_b = c[(c.group == "nonassessable") & (c.slug == "brennan")].iloc[0]
    assert cmp_b.n == na_b.n
    scope = {"all": "All audit stays", "nonassessable": "Non-assessable selected examination"}
    rows = [[scope[r.scope], r.descriptor, n_(r.n), n_(r.deaths), est_ci(r.auroc, r.lo, r.hi, a3), a3(r.brier),
             d2(r.mean_value)] for r in t.itertuples()]
    diffs = "; ".join(
        f"{scope[k].lower()}, {est_ci(s[f'{k}_paired_brennan_minus_motor_mean'], s[f'{k}_paired_brennan_minus_motor_lo'], s[f'{k}_paired_brennan_minus_motor_hi'], d4)}"
        for k in ("all", "nonassessable"))
    return block(
        "**eTable 10. Motor-only severity descriptor in the in-hospital-mortality model (audit subset).** The motor "
        "score alone, which never requires the verbal component, replaced the total GCS in the audit model and was "
        "compared with the Brennan estimate on the same stays. In the second scope the model was fit within the stays "
        "whose selected examination was non-assessable. Mean is the mean motor score for the motor-only rows and the "
        f"mean estimated total GCS for the Brennan rows. AUROC is out-of-fold with a 95% CI from {BOOT} stay-level "
        "bootstrap resamples.",
        table(["Scope", "Descriptor", "Stays, n", "Deaths, n", "AUROC (95% CI)", "Brier", "Mean"], rows,
              ["---", "---"] + ["---:"] * 5),
        f"Paired AUROC difference, Brennan estimate minus motor score only (95% CI): {diffs}. The Brennan AUROC in "
        f"non-assessable stays here ({a3(na_b.auroc)}) comes from a model fit within those stays; eTable 16 gives "
        f"{a3(cmp_b.auroc)} for the same stays from the model fit on all audit stays. A neuromuscular-blocking infusion "
        f"was active at the selected examination in {n_(s['nonassessable_nmb_active_n'])} of the "
        f"{n_(na_b.n)} non-assessable stays ({p1(s['nonassessable_nmb_active_pct'])}%); in those stays the motor "
        "score, and hence the Brennan estimate, cannot be interpreted.")


def t_smd():
    sm_, sel = pd.read_csv(REV / "smd_corrected.csv"), jload("selection_model.json")
    lab = {"Age (years)": "Age, mean, y", "Mechanically ventilated": "Mechanically ventilated, %",
           "Sedative infusion": "Sedative infusion, %", "Vasopressor infusion": "Vasopressor infusion, %",
           "Hospital mortality": "In-hospital mortality, %"}
    rows = [[lab[r.variable], p1(r.excluded), p1(r.retained), d2(r.smd)] for r in sm_.itertuples()]
    n_ex = sel["corrected_n_excluded"]
    return block(
        "**eTable 11. Standardized mean differences between stays excluded and retained by complete-case handling "
        f"(audit subset).** Excluded, non-assessable selected examination (n = {n_(n_ex)}); retained, assessable "
        f"selected examination (n = {n_(sel['corrected_n_stays'] - n_ex)}). Treatments are any infusion or "
        "ventilation episode during the stay. An absolute standardized mean difference above 0.1 indicates imbalance.",
        table(["Characteristic", "Excluded", "Retained", "Standardized mean difference"], rows))


def t_cluster():
    c = jload("cluster_sensitivity.json")
    a4, au = jload("aim4.json"), jload("audit.json")
    rows = []
    for k, lab in [("learned", "Sedation-aware model"), ("learned_no_rass", "Sedation-aware model without RASS"),
                   ("brennan", "Brennan estimate")]:
        rows.append([f"QWK, {lab}", a3(a4[f"{k}_qwk"]),
                     ci(c[f"aim4_stay_{k}_qwk_lo"], c[f"aim4_stay_{k}_qwk_hi"], a3),
                     ci(c[f"aim4_subject_{k}_qwk_lo"], c[f"aim4_subject_{k}_qwk_hi"], a3)])
    rows.append(["QWK difference, model minus Brennan estimate", "",
                 ci(c["aim4_stay_qwk_learned_minus_brennan_lo"], c["aim4_stay_qwk_learned_minus_brennan_hi"], a3),
                 ci(c["aim4_subject_qwk_learned_minus_brennan_lo"], c["aim4_subject_qwk_learned_minus_brennan_hi"], a3)])
    for k in SLUGS:
        rows.append([f"AUROC, {STRATEGY[k]}",
                     a3(au[f"{k}_lowest_em_auroc"]), ci(c[f"audit_stay_{k}_lo"], c[f"audit_stay_{k}_hi"], a3),
                     ci(c[f"audit_subject_{k}_lo"], c[f"audit_subject_{k}_hi"], a3)])
    rows.append(["AUROC difference, Brennan estimate minus official first-day GCS",
                 a3(c["audit_paired_brennan_minus_official_point"]),
                 ci(c["audit_stay_paired_brennan_minus_official_lo"], c["audit_stay_paired_brennan_minus_official_hi"], a3),
                 ci(c["audit_subject_paired_brennan_minus_official_lo"], c["audit_subject_paired_brennan_minus_official_hi"], a3)])
    m = jload("mortality_cv_by_patient.json")
    b_rows = [[STRATEGY[k], a3(m[f"{k}_auroc_stay_folds"]), a3(m[f"{k}_auroc_patient_folds"]), d4(m[f"{k}_diff"])]
              for k in SLUGS]
    stay_pt, pat_pt = m["brennan_minus_official_stay_folds"], m["brennan_minus_official_patient_folds"]
    b_rows.append(["AUROC difference, Brennan estimate minus official first-day GCS", a3(stay_pt), a3(pat_pt),
                   d4(pat_pt - stay_pt)])
    return block(
        "**eTable 12. Sensitivity analyses that treat the patient as the unit: resampling and cross-validation folds.** "
        f"The cohort is the first ICU stay of each hospitalization: {n_(c['n_stays_total'])} stays from "
        f"{n_(c['n_subjects_total'])} patients, of whom {n_(c['n_subjects_with_multiple_stays'])} contributed more than "
        f"one stay. The audit subset comprises {n_(c['audit_n_stays'])} stays from {n_(c['audit_n_subjects'])} "
        f"patients, and the secondary model used {n_(c['aim4_n_exams'])} examinations from {n_(c['aim4_n_stays'])} "
        f"stays of {n_(c['aim4_n_subjects'])} patients. Both intervals are percentiles of {n_(c['n_resamples'])} "
        "bootstrap resamples; the patient-clustered version keeps every stay and examination of a resampled patient, and "
        "the point estimates in Panel A are the same under both.",
        "Panel A. Confidence intervals from resampling ICU stays and from resampling patients",
        table(["Quantity", "Estimate", "95% CI, stays resampled", "95% CI, patients resampled"], rows),
        "Panel B. Cross-validation folds of the mortality model assigned by ICU stay or by patient",
        table(["Strategy", "AUROC, folds assigned by ICU stay", "AUROC, folds assigned by patient",
               "Difference, patient folds minus stay folds"], b_rows),
        "QWK, quadratic weighted kappa against the charted verbal score in assessable examinations; AUROC, audit-subset "
        "mortality model; RASS, Richmond Agitation-Sedation Scale.",
        "The mortality model in Panel B is fitted with five-fold cross-validation stratified on in-hospital death. Of "
        f"the {n_(m['n_patients'])} patients in the audit subset, {n_(c['audit_n_subjects_with_multiple_stays'])} had "
        f"more than one stay. With folds assigned by ICU stay, {n_(m['patients_in_multiple_folds_stay_folds'])} of "
        "them had stays in more than one fold; with folds assigned by patient, none did, and each fold held between "
        f"{p1(m['patient_fold_pct_min'])}% and {p1(m['patient_fold_pct_max'])}% of the stays. Assigning the folds by "
        f"patient changed no AUROC by more than {d4(m['max_abs_diff'])}.")


def _wilson(t, grp, win, stat):
    return (f"{n_(t[f'{grp}_{win}_{stat}_n'])} ({p1(t[f'{grp}_{win}_{stat}_pct'])}; "
            f"{ci(t[f'{grp}_{win}_{stat}_lo'], t[f'{grp}_{win}_{stat}_hi'], p1)})")


def t_eicu():
    t, pr = jload("eicu_time_aligned.json"), jload("eicu_probe.json")
    ex = json.loads((V1 / "external_eicu_digest.json").read_text(encoding="utf-8"))
    wins = [("w0_6h", "0 to 6 hours after the anchor (primary)"), ("w1_6h", "1 to 6 hours after"),
            ("w0_12h", "0 to 12 hours after"), ("pre_6h", "6 hours before the anchor, nearest record (reference)")]
    grps = [("all", "All stays"), ("neuro", "Neurological or cardiac-arrest admission")]
    a_rows = [[g_lab, w_lab, n_(t[f"{g}_{w}_n"]), _wilson(t, g, w, "ge12"), _wilson(t, g, w, "eq15")]
              for g, g_lab in grps for w, w_lab in wins]
    b_rows = [[g_lab, n_(t[f"{g}_paired_n"]),
               f"{n_(t[f'{g}_paired_pre_ge12_n'])} ({p1(t[f'{g}_paired_pre_ge12_pct'])})",
               f"{n_(t[f'{g}_paired_post_ge12_n'])} ({p1(t[f'{g}_paired_post_ge12_pct'])})",
               n_(t[f"{g}_paired_pre_only_ge12_n"]), n_(t[f"{g}_paired_post_only_ge12_n"])] for g, g_lab in grps]
    c_rows = []
    for key, lab in [("full_icu", "All stays"), ("neuro_admission_subgroup", "Neurological or cardiac-arrest admission")]:
        s = ex[key]["selection_trap"]
        for grp, g_lab in [("nonassessable", "Any GCS component not scored"), ("assessable", "All components scored")]:
            m, lo, hi = s[f"mortality_{grp}_pct"]
            c_rows.append([lab, g_lab, n_(s[f"n_{grp}"]), n_(s[f"deaths_{grp}"]), est_ci(m, lo, hi, p1)])
    d_rows = []
    for key, lab in [("full_icu", "All stays"), ("neuro_admission_subgroup", "Neurological or cardiac-arrest admission")]:
        s = ex[key]
        m, lo, hi = s["default_to_normal"]["intubated_totalGCS15_pct"]
        d_rows.append([lab, n_(s["n_intubated"]), est_ci(m, lo, hi, p1)])
    return block(
        "**eTable 13. Replication in eICU-CRD.** eICU-CRD has no \"No Response-ETT\" token. The anchor is the first "
        "documented mechanical-ventilation treatment entry, which is not an intubation time. A charted total of 12 or "
        "more requires a verbal score of at least 2, which cannot be observed through an endotracheal tube, but such a "
        "total cannot separate a charting practice from a timing artifact. Percentages are given with Wilson 95% CIs.",
        "Panel A. Nursing-chart GCS total nearest the anchor in each window (one record per stay)",
        table(["Subgroup", "Window", "Stays, n", "Total of 12 or more, n (%; 95% CI)", "Total of 15, n (%; 95% CI)"],
              a_rows, ["---", "---", "---:", "---:", "---:"]),
        "Panel B. Stays with both a record in the 6 hours before the anchor and a record in the 6 hours after it",
        table(["Subgroup", "Stays, n", "Total of 12 or more before, n (%)", "Total of 12 or more after, n (%)",
               "12 or more before only, n", "12 or more after only, n"], b_rows),
        "Panel C. In-hospital mortality by whether the APACHE record scored every GCS component (not timed to "
        "ventilation)",
        table(["Subgroup", "GCS in the APACHE record", "Stays, n", "Deaths, n", "Mortality, % (95% CI)"], c_rows,
              ["---", "---", "---:", "---:", "---:"]),
        "Panel D. Total GCS of 15 in the APACHE record among stays with the APACHE intubated flag (untimed; the flag "
        "refers to the time of the worst arterial blood gas, not to the GCS examination, and the denominator differs "
        "from Panel A)",
        table(["Subgroup", "Stays with the intubated flag, n", "Total of 15, % (95% CI)"], d_rows),
        f"Windows were applied to minute offsets from the anchor and are inclusive; labels are in hours. The post-anchor windows take the record nearest the "
        f"anchor and the reference takes the latest record in the 6 hours before it. Of {n_(pr['n_gcs_rows_total'])} GCS "
        f"total rows, {n_(pr['n_gcs_rows_nonnumeric'])} had a blank or non-numeric value and were dropped, and "
        f"{n_(pr['n_gcs_rows_numeric_outside_3_15'])} numeric values fell outside 3 to 15. The anchor could be set for "
        f"{n_(pr['n_stays_with_vent'])} stays; in {n_(pr['n_stays_first_vent_row_not_invasive'])} the first ventilation "
        f"entry was tagged non-invasive, and in {n_(pr['n_stays_vent_offset_negative'])} "
        f"({n_(pr['n_stays_vent_offset_negative_gcs_window'])} with a record in the primary window) it preceded unit "
        f"admission. The median anchor was {pr['median_vent_offset_min_all_anchored']:.0f} minutes after unit admission "
        f"({pr['median_vent_offset_min_gcs_window']:.0f} minutes among stays with a record in the primary window).")


def t_selection():
    o = pd.read_csv(REV / "selection_model_corrected.csv")
    sel = jload("selection_model.json")
    lab = {"vent": "Invasive mechanical ventilation", "sed": "Sedative infusion", "vaso": "Vasopressor infusion",
           "died": "In-hospital death", "age": "Age, per year", "ICH vs AIS": "ICH vs AIS", "SAH vs AIS": "SAH vs AIS",
           "SDH vs AIS": "SDH vs AIS", "TBI vs AIS": "TBI vs AIS", "anoxic vs AIS": "Anoxic vs AIS"}
    order = ["vent", "sed", "died", "vaso", "age", "ICH vs AIS", "SAH vs AIS", "SDH vs AIS", "TBI vs AIS", "anoxic vs AIS"]
    o = o.set_index("predictor").loc[order]
    rows = [[lab[k], f"{or_(r.OR)} ({or_(r.lo)} to {or_(r.hi)})", pval(r.p)] for k, r in o.iterrows()]
    return block(
        "**eTable 14. Characteristics associated with a non-assessable selected examination (multivariable logistic "
        f"regression, audit subset, n = {n_(sel['corrected_n_stays'])} stays, {n_(sel['corrected_n_excluded'])} with a "
        "non-assessable selected examination).** Treatments are any infusion or ventilation episode during the stay. "
        "Because an endotracheal tube prevents a verbal response, the association with ventilation is expected and is "
        "not a causal estimate. " + ABBR,
        table(["Characteristic", "Odds ratio (95% CI)", "P value"], rows))


def t_official():
    fe, og, au = jload("feasible.json"), jload("official_gcs.json"), jload("audit.json")
    ca = jload("cohort_accounting.json")
    assert abs(100 * fe["official_15_any_ett_n"] / fe["any_ett_n"] - fe["any_ett_official_15_pct"]) < 1e-9
    equal = og["n_compared"] - og["official_lower_n"] - og["official_higher_n"]
    assert abs(100 * equal / og["n_compared"] - og["agree_pct"]) < 1e-9
    a_rows = [
        [f"**Eligible stays (n = {n_(ca['neuro_first_stay_n'])})**", ""],
        ["With an official first-day GCS", n_(fe["official_n"])],
        ["Official first-day GCS of 15, of stays with one", f"{n_(fe['official_15_n'])} ({p1(fe['official_15_pct'])})"],
        ["Of these, with a non-assessable entry in the first-day window", n_(fe["official_15_any_ett_n"])],
        ["Of these, without a non-assessable entry in the first-day window", n_(fe["official_15_no_ett_n"])],
        ["Stays with a non-assessable entry in the first-day window", n_(fe["any_ett_n"])],
        ["Official first-day GCS of 15, of these", f"{n_(fe['official_15_any_ett_n'])} ({p1(fe['any_ett_official_15_pct'])})"],
        ["Official first-day GCS of 15 assigned by the default rule", n_(fe["official_default_15_n"])],
        ["Charted eye-plus-motor sum below 10 (a total of 15 impossible), of these",
         f"{n_(fe['official_default_15_impossible_n'])} ({p1(fe['official_default_15_impossible_pct'])})"],
        [f"**Audit subset (n = {n_(og['n_compared'])})**", ""],
        ["Non-assessable selected examinations", n_(fe["lowest_em_na_n"])],
        ["Eye-plus-motor sum below 10, of these",
         f"{n_(fe['lowest_em_na_impossible_n'])} ({p1(fe['lowest_em_na_impossible_pct'])})"],
        ["Official first-day minimum equal to the default-to-15 rule's first-day minimum",
         f"{n_(equal)} ({p1(og['agree_pct'])})"],
        ["Official lower", n_(og["official_lower_n"])],
        ["Official higher", n_(og["official_higher_n"])],
        ["Mean APACHE II GCS points, official first-day GCS", d2(au["apache_pts_official"])],
        ["Mean APACHE II GCS points, Brennan estimate", d2(au["apache_pts_brennan"])]]
    assert og["n_compared"] == au["sofa_n"] == ca["aim3_n"]
    ct = pd.read_csv(REV / "sofa_crosstab.csv", index_col="brennan_estimate")
    b_rows = [[str(i)] + [n_(v) for v in ct.loc[i].values] + [n_(ct.loc[i].sum())] for i in ct.index]
    b_rows.append(["Total"] + [n_(ct[c].sum()) for c in ct.columns] + [n_(ct.values.sum())])
    assert int(ct.values.sum()) == au["sofa_n"]
    em = pd.read_csv(REV / "feasible_em_distribution.csv")
    assert int(em.n.sum()) == fe["lowest_em_na_n"]
    c_rows = [[str(r.em), n_(r.n), "Yes" if r.default_15_feasible else "No"] for r in em.itertuples()]
    # Panel B cells above the diagonal: a lower (more severe) category with the official GCS than with the Brennan
    # estimate. Rows are the Brennan category, columns the official category, both in order 0..4.
    above = int(sum(ct.iloc[i, j] for i in range(5) for j in range(5) if j > i))
    return block(
        "**eTable 15. Official MIMIC Code Repository first-day GCS, feasible-range counts, and severity categories "
        "computed with the official GCS and with the Brennan estimate.** The official first-day GCS is the lowest "
        "total from 6 hours before to 24 hours after ICU admission, computed by gcs.sql and first_day_gcs.sql "
        "(eMethods S1.7). Because the verbal score ranges from 1 to 5, a total of 15 requires an eye-plus-motor sum "
        "of 10.",
        "Panel A. Official first-day GCS and totals of 15. Values are stays, n (%), except the mean APACHE II points.",
        table(["Measure", "Value"], a_rows, ["---", "---:"]),
        f"Panel B. SOFA central nervous system category, first-day minimum (n = {n_(au['sofa_n'])} audit stays). Rows, "
        "category with the Brennan estimate (an estimate, not an observed score); columns, category with the official "
        "first-day GCS. Categories: 0, GCS 15; 1, 13 to 14; 2, 10 to 12; 3, 6 to 9; 4, below 6.",
        table(["Brennan estimate \\ official", "0", "1", "2", "3", "4", "Total"], b_rows),
        "Panel C. Eye-plus-motor sum of the non-assessable selected examinations",
        table(["Eye-plus-motor sum", "Examinations, n", "Total of 15 possible"], c_rows, ["---", "---:", "---"]),
        "APACHE II GCS points are 15 minus the first-day minimum total. The default-to-15 rule's first-day minimum is "
        "the lowest total over complete same-time examinations in the first 24 hours with a total of 15 at each "
        "non-assessable examination. The Brennan first-day minimum uses complete same-time examinations from 0 to 24 "
        "hours after ICU admission, whereas the official first-day GCS covers 6 hours before to 24 hours after "
        f"admission and carries components forward; these differences account for the {n_(above)} stays above the "
        "diagonal in Panel B. SOFA, Sequential Organ Failure Assessment.")


def t_calibration():
    c = pd.read_csv(REV / "comparability.csv")
    au = jload("audit.json")
    grp = [("all", "All audit stays"), ("nonassessable", "Non-assessable selected examination"),
           ("assessable", "Assessable selected examination")]
    rows = []
    for g, g_lab in grp:
        for s in SLUGS:
            r = c[(c.group == g) & (c.slug == s)].iloc[0]
            rows.append([g_lab, STRATEGY[s], n_(r.n), n_(r.deaths), p1(r.mortality_pct), p1(r.mean_predicted_pct),
                         d2(r.oe_ratio), d2(r.calib_slope), est_ci(r.auroc, r.lo, r.hi, a3)])
    cc = c[(c.group == "cc") & (c.slug == "complete_case")].iloc[0]
    b_rows = [["Complete-case", n_(cc.n), n_(cc.deaths), p1(cc.mortality_pct), p1(cc.mean_predicted_pct),
               d2(cc.oe_ratio), d2(cc.calib_slope), est_ci(cc.auroc, cc.lo, cc.hi, a3)]]
    return block(
        "**eTable 16. Calibration and discrimination of the in-hospital-mortality model by subgroup of the selected "
        "examination (audit subset).** One model per handling rule was fit on all audit stays; its out-of-fold "
        "predictions were then summarized within each subgroup. Observed-to-expected ratio: observed mortality "
        "divided by mean predicted mortality (above 1, under-prediction). Calibration slope: coefficient of a logistic "
        "regression of death on the log-odds of the prediction.",
        "Panel A. Handling rules applied to the same stays",
        table(["Subgroup", "Handling rule", "Stays, n", "Deaths, n", "Observed mortality, %", "Mean predicted, %",
               "Observed-to-expected ratio", "Calibration slope", "AUROC (95% CI)"], rows,
              ["---", "---"] + ["---:"] * 7),
        "Panel B. Complete-case model, fit on a different population with a different mortality (not comparable "
        "with Panel A)",
        table(["Model", "Stays, n", "Deaths, n", "Observed mortality, %", "Mean predicted, %",
               "Observed-to-expected ratio", "Calibration slope", "AUROC (95% CI)"], b_rows),
        "Over all audit stays, the observed-to-expected ratio is close to 1 for every rule, because out-of-fold "
        "predictions from a model with an intercept reproduce the overall mortality; the subgroup rows are the "
        f"informative comparison. Between fixed imputation and the Brennan estimate, {p1(au['reclass_impute1_vs_brennan_pct'])}% "
        f"of stays changed tertile of predicted risk. AUROC intervals are from {BOOT} stay-level bootstrap resamples.")


def t_rass():
    a4, ca = jload("aim4.json"), jload("cohort_accounting.json")
    fe = jload("feasible.json")
    r = pd.read_csv(REV / "rass_missing_by_subgroup.csv").set_index("subgroup")
    assert r.loc["selected_all", "n"] == ca["aim3_n"] and r.loc["selected_nonassess=True", "n"] == fe["lowest_em_na_n"]
    for k, key in [("all", "rass_missing_all_pct"), ("selected_all", "rass_missing_selected_pct")]:
        assert abs(r.loc[k, "rass_missing_pct"] - a4[key]) < 1e-9

    def row(lab, k):
        return [lab, n_(r.loc[k, "n"]), f"{n_(r.loc[k, 'rass_missing_n'])} ({p1(r.loc[k, 'rass_missing_pct'])})"]
    a_rows = [row("All selected examinations", "selected_all"),
              row("Non-assessable selected examinations", "selected_nonassess=True"),
              row("Assessable selected examinations", "selected_nonassess=False"),
              row("All examinations of the first 72 hours", "all")]
    groups = [("nonassess=False", "Assessable"), ("nonassess=True", "Non-assessable"),
              ("vent=0", "No active ventilation episode"), ("vent=1", "Active ventilation episode")] + PHENO
    b_rows = [row(lab, k) for k, lab in groups]
    return block(
        "**eTable 17. Missing Richmond Agitation-Sedation Scale (RASS) values.** RASS was the most recent value "
        "charted at or before the examination and within the preceding 2 hours; when none was charted in that "
        "interval, RASS was missing. RASS entered only the secondary model, which handles missing values natively (eMethods S1.5).",
        "Panel A. Examinations selected for the audit and all examinations of the first 72 hours",
        table(["Examinations", "Examinations, n", "RASS missing, n (%)"], a_rows),
        "Panel B. All examinations of the first 72 hours, by subgroup",
        table(["Subgroup", "Examinations, n", "RASS missing, n (%)"], b_rows),
        "Ventilation refers to an active invasive-ventilation episode at the time of the examination. " + ABBR)


def t_pheno_or():
    ca = jload("cohort_accounting.json")
    st = json.loads((V1 / "stats_digest.json").read_text(encoding="utf-8"))["aim1_adj_or"]
    by = {r["term"]: r for r in st}
    rows = []
    for k, lab in PHENO:
        if k == "AIS":
            continue
        r = by[f"{k} vs AIS"]
        rows.append([lab, est_ci(r["OR"], r["lo"], r["hi"], or_)])
    assert len(rows) == len(st) == 5
    return block(
        "**eTable 18. Adjusted odds ratios of ever having a non-assessable verbal examination, by phenotype.** "
        "Logistic regression adjusted for age and sex; acute ischemic stroke is the reference; first ICU stay of each "
        f"hospitalization (n = {n_(ca['with_verbal_n'])} stays). " + ABBR,
        table(["Phenotype (reference: acute ischemic stroke)", "Adjusted odds ratio (95% CI)"], rows))


TABLES = {1: t_codes, 2: t_overlap, 3: t_excluded, 4: t_values, 5: t_broadened, 6: t_identifiers, 7: t_rules,
          8: t_brennan, 9: t_model, 10: t_motor, 11: t_smd, 12: t_cluster, 13: t_eicu, 14: t_selection,
          15: t_official, 16: t_calibration, 17: t_rass, 18: t_pheno_or}


def render(text, tables=None):
    tables = tables or {n: f() for n, f in TABLES.items()}
    for n, body in tables.items():
        b, e = BEGIN.format(n=n), END.format(n=n)
        pat = re.compile(re.escape(b) + r"\n.*?" + re.escape(e), re.S)
        if len(pat.findall(text)) != 1:
            raise ValueError(f"expected exactly one marker block for eTable {n}")
        text = pat.sub(lambda _: f"{b}\n\n{body}\n\n{e}", text)
    extra = set(map(int, re.findall(r"<!-- BEGIN eTable (\d+) -->", text))) - set(tables)
    if extra:
        raise ValueError(f"marker blocks with no generator: {sorted(extra)}")
    return text


def blocks(text):
    """{N: text between the markers} for every eTable marker block."""
    return {int(n): body for n, body in
            re.findall(r"<!-- BEGIN eTable (\d+) -->\n(.*?)\n<!-- END eTable \1 -->", text, re.S)}


def main(path=SUPPLEMENT):
    text = path.read_text(encoding="utf-8")
    new = render(text)
    path.write_text(new, encoding="utf-8")
    print(f"wrote {len(blocks(new))} eTable blocks to {path}")


if __name__ == "__main__":
    from pathlib import Path
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else SUPPLEMENT)
