# Handling of the Non-Assessable Verbal Glasgow Coma Scale and Computed Illness Severity in Critically Ill Adults With Acute Brain Injury: A Retrospective Analysis

Gorenshtein A, Adiniaev Y, Omar M, Barash Y, Klang E, Daniel O. Department of Neurology and BRIDGE GenAI Lab, Beth Israel Deaconess Medical Center, Harvard Medical School, Boston, MA, USA. Corresponding author: Alon Gorenshtein, MD (agorensh@bidmc.harvard.edu).

Repository: https://github.com/Alon-Gorenshtein/The_unexaminable_brain

Background: The Glasgow Coma Scale (GCS) contributes to illness-severity scores. The verbal component cannot be observed in intubated patients; the MIMIC Code Repository derivation assigns them a total of 15.

Objective: To quantify non-assessable verbal GCS examinations in adults with acute brain injury and describe how their handling changes computed severity, the analyzed population and mortality-model performance.

Conclusion: In this cohort, how a non-assessable verbal GCS was handled changed computed severity and the analyzed population. Studies and benchmarks that use GCS-based severity should report the handling rule.

## Data

This study uses the **MIMIC-IV v3.1** database, available to credentialed users from PhysioNet:

https://physionet.org/content/mimiciv/3.1/

Raw data are **not** included in this repository and cannot be redistributed under the PhysioNet data use agreement. Obtain credentialed access, download the dataset, and set the local data path in `code/config.py` (replace the `/path/to/mimic-iv-3.1` placeholder). 

## Reproducing the analysis

1. Clone this repository: `git clone https://github.com/Alon-Gorenshtein/The_unexaminable_brain.git`
2. Use Python 3.9 or later and install the packages listed in `code/README.md` (script 60 needs its own environment with `duckdb`, described there).
3. Point the data-path variables at the top of `code/config.py` to your local copies of MIMIC-IV and eICU-CRD.
4. Run the scripts in the order given in `code/README.md`, not in plain numeric order. Intermediate and final outputs are written to `./output/`.

## Repository contents

- `code/` — the full analysis pipeline (cohort construction, extraction, statistics, and figures). Local file-system paths have been replaced with `/path/to/...` placeholders.
- a BibTeX/AMA reference file is included.

## Citation

Gorenshtein A, Adiniaev Y, Omar M, Barash Y, Klang E, Daniel O. Handling of the Non-Assessable Verbal Glasgow Coma Scale and Computed Illness Severity in Critically Ill Adults With Acute Brain Injury: A Retrospective Analysis. 2026.

## License

The code is released under an open-source license on publication.
