import numpy as np

from rng_constants import BOOT_B, BOOT_SEED

LABELS = (1, 2, 3, 4, 5)
SEED = BOOT_SEED
N_RESAMPLES = BOOT_B


def patient_confusions(y, pred, groups, labels=LABELS):
    idx = {l: i for i, l in enumerate(labels)}
    yi = np.fromiter((idx[v] for v in y), dtype=int, count=len(y))
    pi = np.fromiter((idx[v] for v in pred), dtype=int, count=len(pred))
    _, inv = np.unique(groups, return_inverse=True)
    M = np.zeros((inv.max() + 1, len(labels), len(labels)), dtype=np.int64)
    np.add.at(M, (inv, yi, pi), 1)
    return M


def qwk_from_confusion(C):
    C = np.asarray(C, dtype=float)
    K, n = C.shape[0], C.sum()
    i, j = np.indices((K, K))
    W = (i - j) ** 2 / (K - 1) ** 2
    E = np.outer(C.sum(1), C.sum(0)) / n
    return 1.0 - (W * C).sum() / (W * E).sum()


def acc_from_confusion(C):
    C = np.asarray(C, dtype=float)
    return np.trace(C) / C.sum()


def _weights(rng, P):
    return np.bincount(rng.integers(0, P, P), minlength=P)


def cluster_ci(M, stat, B=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    out = [stat(np.tensordot(_weights(rng, M.shape[0]), M, axes=1)) for _ in range(B)]
    return tuple(np.percentile(out, [2.5, 97.5]))


def cluster_diff_ci(M1, M2, stat, B=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        w = _weights(rng, M1.shape[0])
        out.append(stat(np.tensordot(w, M1, axes=1)) - stat(np.tensordot(w, M2, axes=1)))
    return tuple(np.percentile(out, [2.5, 97.5]))


def exam_level_ci(C, stat, B=2000, seed=1):
    rng = np.random.default_rng(seed)
    n, pr = int(C.sum()), (C / C.sum()).ravel()
    out = [stat(rng.multinomial(n, pr).reshape(C.shape)) for _ in range(B)]
    return tuple(np.percentile(out, [2.5, 97.5]))


# ---- cluster bootstrap of the AUROC: resample clusters (patients) with replacement, keep all their rows ----
def _weighted_auc(gid, y, w):
    """AUROC of a sample in which row i is repeated w[i] times. `gid` is the rank of each row's score among
    the distinct scores (np.unique(..., return_inverse=True)[1]), so tied scores count one half."""
    ng = int(gid.max()) + 1
    P = np.bincount(gid, weights=w * y, minlength=ng)
    N = np.bincount(gid, weights=w * (1.0 - y), minlength=ng)
    wp, wn = P.sum(), N.sum()
    if wp == 0 or wn == 0:
        return np.nan
    return float((P * (np.cumsum(N) - N + 0.5 * N)).sum() / (wp * wn))


def _auc_draws(y, scores, groups, B, seed):
    """B x len(scores) array of AUROCs. Every column sees the same resampled patients in every draw."""
    y = np.asarray(y).astype(float)
    _, inv = np.unique(groups, return_inverse=True)
    P = int(inv.max()) + 1
    gids = [np.unique(np.asarray(s), return_inverse=True)[1] for s in scores]
    rng = np.random.default_rng(seed)
    out = np.empty((B, len(scores)))
    for b in range(B):
        w = _weights(rng, P)[inv].astype(float)
        out[b] = [_weighted_auc(g, y, w) for g in gids]
    return out[~np.isnan(out).any(axis=1)]


def cluster_auc_ci(y, p, groups, B=N_RESAMPLES, seed=SEED):
    """95% percentile interval of the AUROC when whole clusters (groups) are resampled."""
    d = _auc_draws(y, [p], groups, B, seed)[:, 0]
    return tuple(np.percentile(d, [2.5, 97.5]))


def cluster_auc_diff_ci(y, pa, pb, groups, B=N_RESAMPLES, seed=SEED):
    """(mean, 2.5th, 97.5th percentile) of AUROC(pa) - AUROC(pb) on the same resampled clusters."""
    d = _auc_draws(y, [pa, pb], groups, B, seed)
    d = d[:, 0] - d[:, 1]
    return (float(np.mean(d)), *np.percentile(d, [2.5, 97.5]))
