"""
K-nearest neighbours written from scratch.

One compiled, multi-threaded kernel performs an exact search over all training rows for four
distance functions. A query keeps the k best candidates in a sorted list, so no distance matrix
is ever stored. A partial sum is abandoned as soon as it exceeds the current k-th best distance.

metric       distance between rows q and t (numeric part n, nominal part c, p features)
euclidean    sqrt( sum_j (q_j - t_j)^2 )                          one-hot encoded input
manhattan    sum_j |q_j - t_j|                                    one-hot encoded input
heom         sqrt( sum_j d_j^2 )   
gower        (1/p) sum_j d_j      
             with d_j = min(1, |q_j - t_j| / range_j) for a numeric feature and
                  d_j = 0 if q_j = t_j else 1                     for a nominal feature

Votes are computed from the stored neighbour lists, so k and the weighting scheme change without
a new search.
"""
import numpy as np
from numba import njit, prange

N_CLASSES = 10
MODES = {"euclidean": 0, "manhattan": 1, "heom": 2, "gower": 3}


@njit(cache=True, parallel=True)
def _search(Qn, Qc, Tn, Tc, inv, k, mode, clip):
    nq, pn = Qn.shape
    nt = Tn.shape[0]
    pc = Qc.shape[1]
    idx = np.empty((nq, k), np.int32)
    dist = np.empty((nq, k), np.float32)
    for i in prange(nq):
        bd = np.full(k, np.inf)
        bi = np.zeros(k, np.int32)
        for t in range(nt):
            worst = bd[k - 1]
            s = 0.0
            for j in range(pn):
                d = abs(np.float64(Qn[i, j]) - np.float64(Tn[t, j])) * inv[j]
                if clip and d > 1.0:
                    d = 1.0
                if mode == 1 or mode == 3:
                    s += d
                else:
                    s += d * d
                if s >= worst:
                    break
            if s < worst:
                for j in range(pc):
                    if Qc[i, j] != Tc[t, j]:
                        s += 1.0
            if s < worst:
                pos = k - 1
                while pos > 0 and bd[pos - 1] > s:
                    bd[pos] = bd[pos - 1]
                    bi[pos] = bi[pos - 1]
                    pos -= 1
                bd[pos] = s
                bi[pos] = t
        p_all = pn + pc
        for q in range(k):
            idx[i, q] = bi[q]
            if mode == 0 or mode == 2:
                dist[i, q] = np.sqrt(bd[q])
            elif mode == 3:
                dist[i, q] = bd[q] / p_all
            else:
                dist[i, q] = bd[q]
    return idx, dist


def knn_search(Q, T, k, metric="euclidean", Qc=None, Tc=None, inv_range=None, clip=False):
    """Indices and distances of the k nearest rows of T for every row of Q (ascending order).

    Q, T      numeric parts (one-hot encoded rows for euclidean and manhattan)
    Qc, Tc    integer codes of the nominal features (heom and gower only)
    inv_range 1 / range of every numeric feature (heom and gower only)
    """
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    T = np.ascontiguousarray(T, dtype=np.float32)
    if Qc is None:
        Qc = np.zeros((len(Q), 0), np.int32)
        Tc = np.zeros((len(T), 0), np.int32)
    inv = np.ones(Q.shape[1]) if inv_range is None else np.asarray(inv_range, dtype=np.float64)
    return _search(Q, np.ascontiguousarray(Qc, dtype=np.int32), T, np.ascontiguousarray(Tc, dtype=np.int32),
                   inv, int(k), MODES[metric], bool(clip))


def votes(idx, dist, y_train, k, weighting="uniform", eps=1e-6):
    """Class score matrix (n_queries x classes) from the first k neighbours."""
    lab = y_train[idx[:, :k]]
    w = np.ones(lab.shape) if weighting == "uniform" else 1.0 / (dist[:, :k] + eps)
    S = np.zeros((len(idx), N_CLASSES))
    for c in range(N_CLASSES):
        S[:, c] = (w * (lab == c)).sum(axis=1)
    return S / S.sum(axis=1, keepdims=True)
