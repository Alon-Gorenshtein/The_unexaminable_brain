import numpy as np
from sklearn.metrics import cohen_kappa_score

import lib_cluster as L


def test_qwk_from_confusion_equals_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(1, 6, 3000)
    p = np.clip(y + rng.integers(-1, 2, 3000), 1, 5)
    g = rng.integers(0, 300, 3000)
    C = L.patient_confusions(y, p, g).sum(0)
    assert abs(L.qwk_from_confusion(C) - cohen_kappa_score(y, p, weights="quadratic", labels=[1, 2, 3, 4, 5])) < 1e-9
    assert abs(L.acc_from_confusion(C) - (y == p).mean()) < 1e-12


def test_cluster_interval_is_wider_when_patients_differ():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(200), 30)
    good = rng.random(200) < 0.5
    y = rng.integers(1, 6, len(g))
    p = np.where(rng.random(len(g)) < np.where(good[g], 0.95, 0.15), y, rng.integers(1, 6, len(g)))
    M = L.patient_confusions(y, p, g)
    lo_c, hi_c = L.cluster_ci(M, L.acc_from_confusion, B=500)
    lo_e, hi_e = L.exam_level_ci(M.sum(0), L.acc_from_confusion, B=500)
    assert (hi_c - lo_c) > 2 * (hi_e - lo_e)
