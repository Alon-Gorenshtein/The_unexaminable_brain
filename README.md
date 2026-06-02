# Non-Assessable Verbal Glasgow Coma Scale Components in Acute Brain Injury: Frequency, Selection Bias, and Severity-Score Distortion in MIMIC-IV

Gorenshtein A, Adiniaev Y, Omar M, Barash Y, Klang E, Daniel O. BRIDGE GenAI Lab, Beth Israel Deaconess Medical Center, Boston, MA, USA.

Background: The verbal component of the Glasgow Coma Scale (GCS) cannot be assessed in intubated patients, yet total GCS values are widely reused in electronic-health-record (EHR) studies and severity scores. How often the verbal component is non-assessable, and how its handling distorts severity scoring, is unknown.

Conclusions: Non-assessable verbal examinations are common and informative, and the convention used to handle them distorts severity scores and mortality models; the MIMIC default-to-15 rule falsely normalizes the sickest patients. EHR studies should report how non-assessable verbal examinations are handled.

## Data

This study uses the **MIMIC-IV v3.1** database, available to credentialed users from PhysioNet:

https://physionet.org/content/mimiciv/3.1/

Raw data are **not** included in this repository and cannot be redistributed under the PhysioNet data use agreement. Obtain credentialed access, download the dataset, and set the local data path in `code/config.py` (replace the `/path/to/mimic-iv-3.1` placeholder). 

## Reproducing the analysis

1. Use Python 3.9 or later. Install the scientific stack: `pandas numpy scipy scikit-learn statsmodels` (and `lifelines`, `torch` where the scripts require them).
2. Point the data-path variable at the top of `code/config.py` to your local copy of MIMIC-IV.
3. Run the scripts in `code/` in numeric order (`00_*`, `01_*`, ...). Intermediate and final outputs are written to `./output/`.

## Repository contents

- `code/` — the full analysis pipeline (cohort construction, extraction, statistics, and figures). Local file-system paths have been replaced with `/path/to/...` placeholders.
- a BibTeX/AMA reference file is included.

## Citation

Gorenshtein A, Adiniaev Y, Omar M, Barash Y, Klang E, Daniel O. Non-Assessable Verbal Glasgow Coma Scale Components in Acute Brain Injury: Frequency, Selection Bias, and Severity-Score Distortion in MIMIC-IV. 2026.

## License

The code is released under an open-source license on publication.
