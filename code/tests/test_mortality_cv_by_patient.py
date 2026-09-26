import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from lib_audit import CV_SEED, cv_splits


def _y(n=300, seed=1):
    return np.random.default_rng(seed).integers(0, 2, n)


def test_without_groups_the_split_is_the_one_used_before():
    y = _y()
    want = list(StratifiedKFold(5, shuffle=True, random_state=CV_SEED).split(np.zeros((len(y), 1)), y))
    got = list(cv_splits(y))
    assert len(got) == 5 and all((a[0] == b[0]).all() and (a[1] == b[1]).all() for a, b in zip(got, want))


def test_with_groups_no_group_is_in_a_training_and_a_test_fold():
    y = _y()
    g = np.repeat(np.arange(100), 3)               # 100 patients, 3 stays each
    for tr, te in cv_splits(y, g):
        assert not set(g[tr]) & set(g[te])


def test_a_leaking_split_is_detected_by_the_same_predicate():
    y = _y()
    g = np.repeat(np.arange(100), 3)
    leaky = list(cv_splits(y))                     # stay-level folds on grouped data
    assert any(set(g[tr]) & set(g[te]) for tr, te in leaky)


# ---- fit_oof: the default is untouched, and `groups` reaches the splitter ----
def _frame(n_patients=120, stays=3, seed=5):
    import pandas as pd
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_patients), stays)
    u = rng.normal(0, 1.5, n_patients)[g]                     # patient effect shared by a patient's stays
    y = (rng.random(len(g)) < 1 / (1 + np.exp(-u))).astype(int)
    w = pd.DataFrame({"stay_id": np.arange(len(g)) + 1000, "y": y, "tg": u + rng.normal(0, 1, len(g)),
                      "age": rng.normal(60, 10, len(g)), "vaso_stay": rng.integers(0, 2, len(g)),
                      "vent_stay": rng.integers(0, 2, len(g))})
    return w, pd.Series(g, index=w["stay_id"])


def test_fit_oof_without_groups_equals_the_stay_fold_fit_used_before():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from lib_audit import fit_oof
    w, _ = _frame()
    ids, y, p = fit_oof(w, "tg")
    d = w.sort_values("stay_id")
    X = d[["tg", "age", "vaso_stay", "vent_stay"]].astype(float).values
    want = np.zeros(len(d))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=CV_SEED).split(X, d["y"].values):
        want[te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], d["y"].values[tr]).predict_proba(X[te])[:, 1]
    assert (ids == d["stay_id"].values).all() and np.array_equal(p, want)


def test_fit_oof_hands_the_splitter_the_patient_of_every_retained_stay(monkeypatch):
    import lib_audit
    w, subject_of = _frame()
    w.loc[w.index[:7], "tg"] = np.nan                         # dropped rows must not shift the mapping
    seen = {}
    real = lib_audit.cv_splits

    def spy(y, groups=None, seed=CV_SEED):
        seen["g"] = groups
        return real(y, groups, seed)
    monkeypatch.setattr(lib_audit, "cv_splits", spy)
    ids, _, _ = lib_audit.fit_oof(w, "tg", groups=subject_of)
    assert np.array_equal(seen["g"], subject_of.loc[ids].values) and len(ids) == len(w) - 7
    lib_audit.fit_oof(w, "tg")
    assert seen["g"] is None


def test_grouped_and_stay_folds_give_different_predictions_on_clustered_data():
    from lib_audit import fit_oof
    w, subject_of = _frame()
    _, _, p_stay = fit_oof(w, "tg")
    _, _, p_pat = fit_oof(w, "tg", groups=subject_of)
    assert not np.allclose(p_stay, p_pat)                     # a fit_oof that ignored `groups` would fail here


# ---- the committed output ----
@pytest.mark.requires_study_files("output/revision/mortality_cv_by_patient.json", "output/revision/audit.json")
def test_output_reproduces_the_stay_fold_auroc_and_has_no_split_patient():
    import json
    from config import REV
    j = json.loads((REV / "mortality_cv_by_patient.json").read_text())
    a = json.loads((REV / "audit.json").read_text())
    slugs = ("official", "reconstructed", "impute1", "gcs_t", "brennan")
    for s in slugs:
        assert abs(j[f"{s}_auroc_stay_folds"] - a[f"{s}_lowest_em_auroc"]) < 1e-9
        assert abs(j[f"{s}_diff"] - (j[f"{s}_auroc_patient_folds"] - j[f"{s}_auroc_stay_folds"])) < 1e-12
    assert j["max_abs_diff"] == max(abs(j[f"{s}_diff"]) for s in slugs)
    assert j["patients_in_multiple_folds_patient_folds"] == 0
    assert j["patients_in_multiple_folds_stay_folds"] > 0     # the stay folds do leak, so the check is not vacuous
    assert 15 < j["patient_fold_pct_min"] <= j["patient_fold_pct_max"] < 25
