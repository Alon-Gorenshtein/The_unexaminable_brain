"""Canonical first-day GCS audit. One place for exam selection, strategy totals, the
cross-validated mortality model and the bootstrap, so every AUROC and interval in the revised
paper comes from the same code path."""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import INT
from lib_gcs_strategies import _kramer_add
from rng_constants import BOOT_B, BOOT_SEED

# numpy 2.0 with macOS Accelerate emits spurious "divide by zero / overflow in matmul" warnings; results are unaffected
warnings.filterwarnings("ignore", message=".*encountered in matmul")

RULES = ("lowest_em", "earliest", "latest")
COVARS = ("age", "vaso_stay", "vent_stay")
CV_SEED = 0


def load_exam_table(legacy=False):
    """All first-72-hour examinations. The corrected table keeps examinations with no eye opening;
    the legacy table (the submitted analysis) lost them to a parsing error."""
    return pd.read_parquet(INT / ("aim4_exams.parquet" if legacy else "aim4_exams_fixed.parquet"))


def load_first_day(hours=24, legacy=False):
    ex = load_exam_table(legacy)
    d = ex[(ex["hr"] >= 0) & (ex["hr"] <= hours)].dropna(subset=["eye", "motor"]).copy()
    d["em"] = d["eye"] + d["motor"]
    return d


def load_first_day_carryforward(hours=24):
    """Examinations as the official derivation sees them: components are carried forward for up to
    6 hours, so a patient whose eye, motor and verbal entries carry different timestamps is kept.
    Needs output/revision/official_gcs_by_time.parquet (written by code/60_official_gcs_oracle.py)."""
    from config import ICU, REV
    t = pd.read_parquet(REV / "official_gcs_by_time.parquet")
    ic = pd.read_csv(ICU / "icustays.csv.gz", usecols=["stay_id", "intime"], parse_dates=["intime"])
    c = pd.read_parquet(INT / "cohort.parquet")
    c = c[c["first_stay"] & (c["phenotype"] != "comparator")][["stay_id", "phenotype", "age", "hospital_expire_flag"]]
    d = t.merge(ic, on="stay_id").merge(c, on="stay_id")
    d["hr"] = (d["charttime"] - d["intime"]).dt.total_seconds() / 3600
    d = d[(d["hr"] >= 0) & (d["hr"] <= hours)].dropna(subset=["gcs_eyes", "gcs_motor", "gcs_verbal"]).copy()
    d["eye"], d["motor"] = d["gcs_eyes"], d["gcs_motor"]
    d["em"] = d["eye"] + d["motor"]
    d["nonassess"] = d["gcs_verbal"] == 0                     # gcs.sql codes No Response-ETT as 0
    d["verbal_score"] = d["gcs_verbal"].where(~d["nonassess"])
    d["y"] = d["hospital_expire_flag"].astype(int)
    return d


def selected_frames(hours=24):
    """The selected examination per stay under every rule, keyed by rule name. The first four frames use
    the corrected population. `legacy_population_lowest_em` is the submitted population (examinations
    with no eye opening lost to a parsing error) and exists only to reconcile with the submitted numbers."""
    frames = {r: select_exam(load_first_day(hours), r) for r in RULES}
    frames["carryforward_lowest_em"] = select_exam(load_first_day_carryforward(hours), "lowest_em")
    frames["legacy_population_lowest_em"] = select_exam(load_first_day(hours, legacy=True), "lowest_em")
    return frames


def select_exam(d, rule):
    """One row per stay. Uses drop_duplicates, NOT groupby().first(): first() takes the first
    non-null value of each column, so a missing RASS on the selected exam would be filled in
    from a different exam."""
    if rule == "lowest_em":
        s = d.sort_values(["stay_id", "em", "charttime"])
    elif rule == "earliest":
        s = d.sort_values(["stay_id", "charttime", "em"])
    elif rule == "latest":
        s = d.sort_values(["stay_id", "charttime", "em"], ascending=[True, False, True])
    else:
        raise ValueError(rule)
    return s.drop_duplicates("stay_id", keep="first").sort_values("stay_id").reset_index(drop=True)


def add_stay_covariates(w):
    inf = pd.read_parquet(INT / "infusions.parquet", columns=["stay_id", "cls"])
    vent = pd.read_parquet(INT / "vent.parquet", columns=["stay_id"])
    w = w.copy()
    w["vaso_stay"] = w["stay_id"].isin(inf.loc[inf["cls"] == "vasopressor", "stay_id"]).astype(int)
    w["vent_stay"] = w["stay_id"].isin(vent["stay_id"]).astype(int)
    return w


def add_totals(w):
    w = w.copy()
    na = w["nonassess"].values.astype(bool)
    em = w["em"].astype(float)
    v = w["verbal_score"].astype(float).where(~w["nonassess"].astype(bool))
    w["tg_derived"] = np.where(na, 15.0, em + v)
    w["tg_impute1"] = np.where(na, em + 1, em + v)
    w["tg_gcs_t"] = np.where(na, em, em + v)
    w["tg_brennan"] = np.where(na, em + em.map(_kramer_add), em + v)
    w["tg_drop"] = np.where(na, np.nan, em + v)
    return w


def cv_splits(y, groups=None, seed=CV_SEED):
    """5-fold splits stratified on the outcome; whole groups (patients) stay together when `groups` is given."""
    X = np.zeros((len(y), 1))
    if groups is None:
        return StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)
    return StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups)


def fit_oof(w, col, covars=COVARS, seed=CV_SEED, groups=None):
    d = w.dropna(subset=[col, *covars]).sort_values("stay_id")
    X = d[[col, *covars]].astype(float).values
    y = d["y"].astype(int).values
    g = None if groups is None else d["stay_id"].map(groups).values
    oof = np.zeros(len(y))
    for tr, te in cv_splits(y, g, seed):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return d["stay_id"].values, y, oof


def boot_auc_ci(y, p, B=BOOT_B, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    n, out = len(y), []
    for _ in range(B):
        s = rng.integers(0, n, n)
        if y[s].sum() in (0, n):
            continue
        out.append(roc_auc_score(y[s], p[s]))
    return tuple(np.percentile(out, [2.5, 97.5]))


def boot_auc_diff(y, pa, pb, B=BOOT_B, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    n, out = len(y), []
    for _ in range(B):
        s = rng.integers(0, n, n)
        if y[s].sum() in (0, n):
            continue
        out.append(roc_auc_score(y[s], pa[s]) - roc_auc_score(y[s], pb[s]))
    return (float(np.mean(out)), *np.percentile(out, [2.5, 97.5]))


def calib_slope(y, p):
    """Calibration slope: coefficient on the log-odds of the predicted probability in a logistic fit."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(sm.Logit(y, sm.add_constant(np.log(p / (1 - p)))).fit(disp=0).params[1])
