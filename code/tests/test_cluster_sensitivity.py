import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import lib_audit as A
import lib_cluster as L
from config import INT, REV

PATH = REV / "cluster_sensitivity.json"
METHODS = ("learned", "learned_no_rass", "brennan", "fixed1")
SLUGS = ("official", "reconstructed", "impute1", "gcs_t", "brennan")


def _synthetic(shift_sd, n_clusters=120, size=25, seed=3):
    """Outcome and score both shifted by a shared cluster effect when shift_sd > 0."""
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_clusters), size)
    u = rng.normal(0, shift_sd, n_clusters)[g]
    y = (rng.random(len(g)) < 1 / (1 + np.exp(-(u - 0.8)))).astype(int)
    p = 0.8 * y + u + rng.normal(0, 1, len(g))
    return y, p, g


def _width(ci):
    return ci[1] - ci[0]


# ---- (a) the cluster-bootstrap AUROC ----
def test_weighted_auc_equals_sklearn_on_the_expanded_sample_including_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.round(rng.normal(size=400) + y, 1)                      # many ties
    w = rng.integers(0, 4, 400)
    gid = np.unique(p, return_inverse=True)[1]
    got = L._weighted_auc(gid, y.astype(float), w.astype(float))
    rep = np.repeat(np.arange(400), w)
    assert abs(got - roc_auc_score(y[rep], p[rep])) < 1e-12
    assert abs(got - roc_auc_score(y, p, sample_weight=w)) < 1e-12


def test_independent_rows_give_the_row_level_interval():
    y, p, _ = _synthetic(0.0)
    row = A.boot_auc_ci(y, p, B=600)
    clu = L.cluster_auc_ci(y, p, np.arange(len(y)), B=600)
    assert 0.85 < _width(clu) / _width(row) < 1.2
    assert np.allclose(clu, row, atol=1e-12)                       # one cluster per row IS the row-level bootstrap


def test_clustered_rows_widen_the_interval():
    y, p, g = _synthetic(1.5)
    row = A.boot_auc_ci(y, p, B=600)
    clu = L.cluster_auc_ci(y, p, g, B=600)
    assert _width(clu) / _width(row) >= 1.3


def test_paired_difference_uses_the_same_resamples():
    y, p, g = _synthetic(1.0)
    m, lo, hi = L.cluster_auc_diff_ci(y, p, p, g, B=200)         # identical scores: difference is exactly zero
    assert m == 0.0 and lo == 0.0 and hi == 0.0
    m, lo, hi = L.cluster_auc_diff_ci(y, p, -p, g, B=200)        # AUC(p) - AUC(-p) = 2 AUC(p) - 1
    point = roc_auc_score(y, p) - roc_auc_score(y, -p)
    assert lo < point < hi


# ---- (c) determinism ----
def test_same_seed_gives_the_same_interval_and_a_different_seed_does_not():
    y, p, g = _synthetic(1.0)
    a = L.cluster_auc_ci(y, p, g, B=300, seed=7)
    assert a == L.cluster_auc_ci(y, p, g, B=300, seed=7)
    assert a != L.cluster_auc_ci(y, p, g, B=300, seed=8)
    d = L.cluster_auc_diff_ci(y, p, p + np.random.default_rng(1).normal(size=len(p)), g, B=300, seed=7)
    assert d == L.cluster_auc_diff_ci(y, p, p + np.random.default_rng(1).normal(size=len(p)), g, B=300, seed=7)


# ---- (b) the written artefact ----
def _no_constants(c):
    raise ValueError(f"non-strict JSON constant {c}")


@pytest.fixture(scope="module")
def js():
    assert PATH.exists(), "run code/67_cluster_sensitivity.py"
    return json.loads(PATH.read_text(), parse_constant=_no_constants)


@pytest.fixture(scope="module")
def a4():
    return json.loads((REV / "aim4.json").read_text())


@pytest.fixture(scope="module")
def au():
    return json.loads((REV / "audit.json").read_text())


@pytest.mark.requires_study_files("output/revision/cluster_sensitivity.json")
def test_json_is_strict_and_counts_are_integers(js):
    assert all(not isinstance(v, float) or np.isfinite(v) for v in js.values())
    counts = [k for k in js if "_n_" in k or k.startswith("n_") or k in ("max_stays_per_subject", "seed")]
    assert {"max_stays_per_subject", "n_resamples", "seed", "n_subjects_total", "aim4_n_subjects",
            "audit_n_subjects_with_multiple_stays"} <= set(counts)
    for k in counts:
        assert isinstance(js[k], int) and not isinstance(js[k], bool), k


@pytest.mark.requires_study_files("output/revision/cluster_sensitivity.json", "output/revision/aim4.json")
def test_aim4_point_estimates_equal_the_stay_level_run(js, a4):
    for m in METHODS:
        assert abs(js[f"aim4_subject_{m}_qwk"] - a4[f"{m}_qwk"]) < 1e-9
        assert abs(js[f"aim4_subject_{m}_acc"] - a4[f"{m}_acc"]) < 1e-9
        assert js[f"aim4_stay_{m}_qwk_lo"] == a4[f"{m}_qwk_lo"] and js[f"aim4_stay_{m}_qwk_hi"] == a4[f"{m}_qwk_hi"]
    assert js["aim4_stay_qwk_learned_minus_brennan_lo"] == a4["qwk_learned_minus_brennan_lo"]
    assert js["aim4_stay_qwk_learned_minus_brennan_hi"] == a4["qwk_learned_minus_brennan_hi"]
    assert js["aim4_n_exams"] == a4["n_exams"] and js["aim4_n_stays"] == a4["n_patients"]


@pytest.mark.requires_study_files("output/revision/cluster_sensitivity.json", "output/revision/audit.json")
def test_audit_point_estimates_equal_the_stay_level_run(js, au):
    for s in SLUGS:
        assert abs(js[f"audit_subject_{s}_auroc"] - au[f"{s}_lowest_em_auroc"]) < 1e-9
        assert js[f"audit_stay_{s}_lo"] == au[f"{s}_lowest_em_lo"] and js[f"audit_stay_{s}_hi"] == au[f"{s}_lowest_em_hi"]
    point = au["brennan_lowest_em_auroc"] - au["official_lowest_em_auroc"]
    assert abs(js["audit_paired_brennan_minus_official_point"] - point) < 1e-9
    assert js["audit_stay_paired_brennan_minus_official_lo"] == au["paired_brennan_minus_official_lo"]
    assert js["audit_stay_paired_brennan_minus_official_hi"] == au["paired_brennan_minus_official_hi"]
    assert js["audit_n_stays"] == au["paired_brennan_minus_official_n"]


@pytest.mark.requires_study_files("output/revision/cluster_sensitivity.json")
def test_every_subject_level_interval_contains_its_point_estimate(js):
    eps = 1e-12
    for m in METHODS:
        for stat in ("qwk", "acc"):
            assert js[f"aim4_subject_{m}_{stat}_lo"] - eps <= js[f"aim4_subject_{m}_{stat}"] <= js[f"aim4_subject_{m}_{stat}_hi"] + eps
    for s in SLUGS:
        assert js[f"audit_subject_{s}_lo"] <= js[f"audit_subject_{s}_auroc"] <= js[f"audit_subject_{s}_hi"]
    p = js["audit_paired_brennan_minus_official_point"]
    assert js["audit_subject_paired_brennan_minus_official_lo"] <= p <= js["audit_subject_paired_brennan_minus_official_hi"]
    assert js["aim4_subject_qwk_learned_minus_brennan_lo"] <= js["aim4_subject_learned_qwk"] - js["aim4_subject_brennan_qwk"] \
        <= js["aim4_subject_qwk_learned_minus_brennan_hi"]


def _subject_of():
    c = pd.read_parquet(INT / "cohort.parquet", columns=["subject_id", "stay_id", "first_stay", "phenotype"])
    return c[c["first_stay"] & (c["phenotype"] != "comparator")]


def _n_multi(subject_ids):
    return int((pd.Series(np.asarray(subject_ids)).value_counts() > 1).sum())


@pytest.mark.requires_study_files("output/revision/cluster_sensitivity.json", "output/intermediate/cohort.parquet", "output/revision/aim4_oof.parquet", "output/revision/audit_oof_lowest_em.parquet")
def test_subject_counts_are_recomputed_from_the_raw_files(js):
    """Recomputed here from cohort.parquet and the two out-of-fold files, so grouping by stay_id fails."""
    c = _subject_of()
    assert js["n_stays_total"] == len(c)
    assert js["n_subjects_total"] == c["subject_id"].nunique() < len(c)
    assert js["n_subjects_with_multiple_stays"] == _n_multi(c["subject_id"]) > 0
    assert js["max_stays_per_subject"] == int(c.groupby("subject_id").size().max()) > 1
    s = c.set_index("stay_id")["subject_id"]

    a4 = pd.read_parquet(REV / "aim4_oof.parquet", columns=["stay_id"])
    stays = a4["stay_id"].drop_duplicates()
    assert js["aim4_n_exams"] == len(a4) and js["aim4_n_stays"] == len(stays)
    assert js["aim4_n_subjects"] == s.loc[stays].nunique() < len(stays)

    ids = pd.read_parquet(REV / "audit_oof_lowest_em.parquet", columns=["y"]).index
    assert js["audit_n_stays"] == len(ids)
    assert js["audit_n_subjects"] == s.loc[ids].nunique() < len(ids)
    assert js["audit_n_subjects_with_multiple_stays"] == _n_multi(s.loc[ids]) > 0

    # the file is internally consistent: every multi-stay patient contributes at least two stays
    assert js["n_stays_total"] >= js["n_subjects_total"] + js["n_subjects_with_multiple_stays"]
    assert js["aim4_n_subjects"] <= js["aim4_n_stays"] and js["audit_n_subjects"] <= js["audit_n_stays"]


@pytest.mark.requires_study_files("output/revision/cluster_sensitivity.json")
def test_subject_level_endpoints_differ_from_the_stay_level_ones(js):
    """A subject-level interval that equals its stay-level twin means the grouping was stay_id in both."""
    pairs = []
    for m in METHODS:
        if m == "fixed1":
            continue      # fixed1 QWK is 0 in every resample at both levels (a constant prediction has no kappa): degenerate
        pairs += [(f"aim4_subject_{m}_qwk_{e}", f"aim4_stay_{m}_qwk_{e}") for e in ("lo", "hi")]
    for m in METHODS:     # accuracy of a constant prediction still varies with the resample, fixed1 is included
        pairs += [(f"aim4_subject_{m}_acc_{e}", f"aim4_stay_{m}_acc_{e}") for e in ("lo", "hi")]
    for s in SLUGS:
        pairs += [(f"audit_subject_{s}_{e}", f"audit_stay_{s}_{e}") for e in ("lo", "hi")]
    pairs += [(f"audit_subject_paired_brennan_minus_official_{e}", f"audit_stay_paired_brennan_minus_official_{e}")
              for e in ("lo", "hi")]
    pairs += [(f"aim4_subject_qwk_learned_minus_brennan_{e}", f"aim4_stay_qwk_learned_minus_brennan_{e}")
              for e in ("lo", "hi")]
    assert len(pairs) == 6 + 8 + 10 + 2 + 2
    same = [a for a, b in pairs if js[a] == js[b]]
    assert not same, same


@pytest.mark.requires_study_files("output/revision/cluster_sensitivity.json", "output/revision/aim4_oof.parquet")
def test_learned_qwk_subject_interval_is_recomputed_from_the_raw_parquet(js):
    oof = pd.read_parquet(REV / "aim4_oof.parquet")
    oof["subject_id"] = oof["stay_id"].map(_subject_of().set_index("stay_id")["subject_id"])
    assert oof["subject_id"].notna().all()
    M = L.patient_confusions(oof["y"].values, oof["learned"].values, oof["subject_id"].values)
    lo, hi = L.cluster_ci(M, L.qwk_from_confusion, B=L.N_RESAMPLES, seed=L.SEED)
    assert abs(lo - js["aim4_subject_learned_qwk_lo"]) < 1e-9 and abs(hi - js["aim4_subject_learned_qwk_hi"]) < 1e-9
    assert abs(L.qwk_from_confusion(M.sum(0)) - js["aim4_subject_learned_qwk"]) < 1e-9


@pytest.mark.requires_study_files("output/revision/cluster_sensitivity.json", "output/revision/aim4_oof.parquet")
def test_resample_count_and_seed_are_the_library_constants(js):
    assert js["n_resamples"] == L.N_RESAMPLES and js["seed"] == L.SEED
    assert L.SEED == A.BOOT_SEED and L.N_RESAMPLES == A.BOOT_B             # one seed and one B across the audit and Aim 4
    for f in (L.cluster_ci, L.cluster_diff_ci, L.cluster_auc_ci, L.cluster_auc_diff_ci):
        assert inspect.signature(f).parameters["B"].default == L.N_RESAMPLES, f.__name__
        assert inspect.signature(f).parameters["seed"].default == L.SEED, f.__name__
