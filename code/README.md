# Analysis code

This folder holds the complete analysis pipeline. Raw data and every file derived from them (the `output/`
folder) are not included, because MIMIC-IV and eICU-CRD may be shared only with credentialed users under the
PhysioNet data use agreement. Running the analysis writes `output/` again.

## Environment

- Python 3.9 with the pinned packages in `code/requirements.txt`:
  `python3 -m pip install -r code/requirements.txt`. No script imports lifelines or torch.
- Script 60 runs the official MIMIC Code Repository SQL in DuckDB, in a separate environment built from
  `code/requirements-duckdb.txt`. The SQL files and their pinned commit are in `code/vendor/mimic_code/`.
- The document tools in `code/tools/` that read or write .docx files need `code/requirements-docs.txt` and pandoc.

Set `DATA` (MIMIC-IV v3.1) and `EICU_DATA` (eICU-CRD v2.0) at the top of `code/config.py`. Outputs are written
under `output/` relative to the working directory, so run every command from the repository root.

## Order

The scripts do not all run in numeric order (for example, 14 reads the examination table that 15 stages). Use this order.

First-submission analyses (Aims 1-2):

```bash
python3 code/00_build_cohort.py
python3 code/01_extract_chartevents.py
python3 code/02_extract_infusions.py
python3 code/03_extract_vent.py
python3 code/10_aim1_burden.py
python3 code/11_aim2_trajectory.py
python3 code/12_aim3_audit.py
python3 code/13_sensitivity.py
python3 code/15_aim4_learned_imputation.py
python3 code/14_stats.py
python3 code/15a_aim4_ablation.py
python3 code/16_calibration.py
python3 code/17_derived_gcs_audit.py
python3 code/18_selection_timecourse.py
python3 code/20_table1.py
python3 code/40_results_digest.py
```

`30_figures.py` and `31_flow.py` are retained only because they produced the first-submission figures. They are
superseded by `32_figures_final.py` and `33_fig_derived.py`; do not run them.

Revision analyses:

```bash
python3 code/04_fix_eye_none.py
python3 code/50_external_eicu.py
<duckdb-environment>/bin/python code/60_official_gcs_oracle.py
python3 code/61_feasible_range_audit.py
python3 code/62_audit_table.py
python3 code/63_audit_comparability.py
python3 code/64_cohort_accounting.py
python3 code/65_aim4_revision.py
python3 code/66_selection_model.py
python3 code/67_cluster_sensitivity.py
python3 code/76_mortality_cv_by_patient.py
python3 code/68_time_to_first_na.py
python3 code/69_motor_only.py
python3 code/70_eicu_probe.py
python3 code/71_eicu_time_aligned.py
python3 code/32_figures_final.py
python3 code/33_fig_derived.py
python3 code/tools/export_figures.py
python3 code/75_export_supplement_inputs.py
python3 code/72_revision_digest.py
```

`72_revision_digest.py` merges the revision outputs into `output/revision/revision_digest.json`. It also reads
the frozen first-submission digests in `output/v1_submitted/` when that folder is present; it is not part of this
copy.

`76_mortality_cv_by_patient.py` reads the outputs of 04, 60 and 62 and must run before 72. After 72,
`python3 code/tools/verify_headline_counts.py` checks that the counts the paper is built on equal the
regenerated digest.

## What needs the study repository

These scripts build or check the submitted documents. They read the manuscript sources (`manuscript/`), the
frozen first-submission outputs (`output/v1_submitted/`), the claims registry (`output/revision/claims.tsv`) or
`docs/revision/packet_terms.tsv`, none of which are in this copy, so they cannot be run from it:

- `74_build_supplement_tables.py` (writes the supplementary tables into `manuscript/supplement.md`; it also reads
  `output/v1_submitted/external_eicu_digest.json` and the files written by script 75)
- `tools/check_claims.py`, `tools/check_response_quotes.py`, `tools/check_packet.py`, `tools/numbers_diff.py`,
  `tools/lint_language.py`, `tools/strict_wordcount.py`, `tools/mark_changes.py`, `tools/highlight_to_color.py`,
  `tools/add_page_numbers.py`, `tools/style_response_quotes.py`, `tools/set_docx_metadata.py`

## Tests

`python3 -m pytest code/tests -q` from the repository root. A test that needs a file this copy does not contain
(an analysis output written from the credentialed data, a first-submission table, or a manuscript source) is
reported as skipped, and the skip reason names the missing file. After the analysis has been run, the tests on
its outputs run; the tests against the first-submission tables and the manuscript stay skipped. `test_env.py`
checks that the installed versions are the pinned ones.
