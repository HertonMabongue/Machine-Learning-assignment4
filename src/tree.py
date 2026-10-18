"""
Classification tree (CART family) with four split criteria.

Design
------
* Exact search: every threshold between two different feature values is evaluated.
* Presorting: every feature is sorted once. A node owns one contiguous segment of every
  sorted list, and a split partitions the segments stably, so no node ever re-sorts.
* Instance weights: class weights and bootstrap multiplicities enter as weights. A row with
  weight zero is ignored, which is how a bootstrap sample is expressed.
* Random feature subsets: at every node `max_features` features are drawn without replacement,
  as in a random forest. Passing all features gives an ordinary tree. A column that is constant
  within the node cannot split. With count_constants=False such a column does not use up one of
  the `max_features` draws, so `max_features` columns that vary in the node are examined. With
  count_constants=True the draw includes constant columns, as in the textbook description
  and in scikit-learn. Either way, drawing continues until a useful split exists.
* Impurity sums are updated in O(1) per threshold, so one pass over a feature costs O(n).

Criteria (for a binary split of a node into left and right)
gini        : largest decrease of the Gini impurity .
entropy     : largest information gain.
gain_ratio  : per feature the threshold with the largest information gain, then the feature
              with the largest ratio of information gain to split information.
c45         : as gain_ratio, but only features whose information gain is at least the average
              information gain of all examined features compete (the average-gain filter of
              Quinlan, 1993). The filter stops the ratio from favouring features with a tiny
              split information.
"""
import numpy as np
from numba import njit, prange

GINI, ENTROPY, GAIN_RATIO, C45 = 0, 1, 2, 3
CRITERIA = {"gini": GINI, "entropy": ENTROPY, "gain_ratio": GAIN_RATIO, "c45": C45}
NO_LIMIT = 1 << 30


def presort(X):
    """Row indices of every feature in ascending order of the feature value. Shape (p, n)."""
    return np.ascontiguousarray(np.argsort(X, axis=0, kind="stable").T.astype(np.int32))


@njit(cache=True, nogil=True)
def _xlogx(x):
    return x * np.log(x) if x > 0.0 else 0.0


@njit(cache=True, nogil=True)
def _grow(X, y, w, order, n_classes, crit, m_try, min_leaf, min_split, max_depth, count_const, seed):
    """Grow one tree. Returns arrays describing the nodes (feature -1 marks a leaf)."""
    np.random.seed(seed)
    n, p = X.shape
    # rows with positive weight form the training sample of this tree
    n_u = 0
    for i in range(n):
        if w[i] > 0.0:
            n_u += 1
    ordr = np.empty((p, n_u), np.int32)
    for j in range(p):
        c = 0
        for k in range(n):
            r = order[j, k]
            if w[r] > 0.0:
                ordr[j, c] = r
                c += 1
    cap = 2 * (n_u // min_leaf) + 3
    feat = np.full(cap, -1, np.int32)
    thr = np.zeros(cap)
    left = np.zeros(cap, np.int32)
    right = np.zeros(cap, np.int32)
    val = np.zeros((cap, n_classes))
    lo = np.zeros(cap, np.int32)
    hi = np.zeros(cap, np.int32)
    dep = np.zeros(cap, np.int32)
    stack = np.zeros(cap, np.int32)
    gain_sum = np.zeros(p)
    perm = np.arange(p)
    tmp = np.empty(n_u, np.int32)
    goes = np.zeros(n, np.uint8)
    Wc = np.zeros(n_classes)
    cl = np.zeros(n_classes)
    cg = np.zeros(p)       # candidate gain per examined feature
    csi = np.zeros(p)      # candidate split information
    cj = np.zeros(p, np.int32)
    cthr = np.zeros(p)
    cnl = np.zeros(p, np.int32)

    n_nodes = 1
    hi[0] = n_u
    stack[0] = 0
    sp = 1
    w_root = 0.0
    while sp > 0:
        sp -= 1
        node = stack[sp]
        a = lo[node]
        b = hi[node]
        cnt = b - a
        for c in range(n_classes):
            Wc[c] = 0.0
        for k in range(a, b):
            r = ordr[0, k]
            Wc[y[r]] += w[r]
        W = 0.0
        for c in range(n_classes):
            W += Wc[c]
        for c in range(n_classes):
            val[node, c] = Wc[c] / W
        if node == 0:
            w_root = W
        if dep[node] >= max_depth or cnt < min_split or cnt < 2 * min_leaf:
            continue
        wmax = 0.0
        for c in range(n_classes):
            if Wc[c] > wmax:
                wmax = Wc[c]
        if wmax >= W * (1.0 - 1e-12):
            continue                                   # pure node
        # impurity of the node
        sq = 0.0
        ent = 0.0
        for c in range(n_classes):
            sq += Wc[c] * Wc[c]
            ent += _xlogx(Wc[c])
        if crit == GINI:
            imp = 1.0 - sq / (W * W)
        else:
            imp = np.log(W) - ent / W
        # examine features until m_try non-constant ones were seen and a useful split exists
        visited = 0
        n_cand = 0
        i = 0
        while i < p and (visited < m_try or n_cand == 0):
            k2 = i + np.random.randint(0, p - i)
            tswap = perm[i]
            perm[i] = perm[k2]
            perm[k2] = tswap
            j = perm[i]
            i += 1
            if X[ordr[j, a], j] == X[ordr[j, b - 1], j]:
                if count_const == 1:
                    visited += 1                       # textbook rule: a constant column uses up one of the m draws
                continue                               # constant in this node
            visited += 1
            for c in range(n_classes):
                cl[c] = 0.0
            sum_l = 0.0
            sum_r = 0.0
            for c in range(n_classes):
                if crit == GINI:
                    sum_r += Wc[c] * Wc[c]
                else:
                    sum_r += _xlogx(Wc[c])
            sum_l = 0.0
            Wl = 0.0
            fg = 0.0
            fthr = 0.0
            fnl = 0
            fwl = 0.0
            for kk in range(a, b - 1):
                r = ordr[j, kk]
                c = y[r]
                wr = w[r]
                old = cl[c]
                oldr = Wc[c] - old
                if crit == GINI:
                    sum_l += 2.0 * old * wr + wr * wr
                    sum_r += wr * wr - 2.0 * oldr * wr
                else:
                    sum_l += _xlogx(old + wr) - _xlogx(old)
                    sum_r += _xlogx(oldr - wr) - _xlogx(oldr)
                cl[c] = old + wr
                Wl += wr
                vcur = X[r, j]
                vnext = X[ordr[j, kk + 1], j]
                if vcur == vnext:
                    continue
                nl = kk - a + 1
                if nl < min_leaf or cnt - nl < min_leaf:
                    continue
                Wr = W - Wl
                if Wl <= 0.0 or Wr <= 0.0:
                    continue
                if crit == GINI:
                    il = 1.0 - sum_l / (Wl * Wl)
                    ir = 1.0 - sum_r / (Wr * Wr)
                else:
                    il = np.log(Wl) - sum_l / Wl
                    ir = np.log(Wr) - sum_r / Wr
                g = imp - (Wl / W) * il - (Wr / W) * ir
                if g > fg:
                    fg = g
                    fthr = 0.5 * (np.float64(vcur) + np.float64(vnext))
                    fnl = nl
                    fwl = Wl
            if fg > 1e-12:
                pl = fwl / W
                pr = 1.0 - pl
                cg[n_cand] = fg
                csi[n_cand] = -(_xlogx(pl) + _xlogx(pr))
                cj[n_cand] = j
                cthr[n_cand] = fthr
                cnl[n_cand] = fnl
                n_cand += 1
        if n_cand == 0:
            continue
        # choose among the candidates
        best = -1
        if crit == GINI or crit == ENTROPY:
            bg = -1.0
            for q in range(n_cand):
                if cg[q] > bg:
                    bg = cg[q]
                    best = q
        else:
            floor = -1.0
            if crit == C45:
                s = 0.0
                for q in range(n_cand):
                    s += cg[q]
                floor = s / n_cand - 1e-12
            br = -1.0
            for q in range(n_cand):
                if cg[q] >= floor and csi[q] > 1e-12:
                    ratio = cg[q] / csi[q]
                    if ratio > br:
                        br = ratio
                        best = q
        if best < 0:
            continue
        bj = cj[best]
        nl = cnl[best]
        feat[node] = bj
        thr[node] = cthr[best]
        gain_sum[bj] += W * cg[best]
        # children
        lchild = n_nodes
        rchild = n_nodes + 1
        n_nodes += 2
        left[node] = lchild
        right[node] = rchild
        lo[lchild] = a
        hi[lchild] = a + nl
        lo[rchild] = a + nl
        hi[rchild] = b
        dep[lchild] = dep[node] + 1
        dep[rchild] = dep[node] + 1
        # mark the side of every row of the segment (the sorted list of bj puts the left rows first)
        for k in range(a, b):
            goes[ordr[bj, k]] = 1 if k < a + nl else 0
        expand = False
        if dep[node] + 1 < max_depth:
            if (nl >= 2 * min_leaf and nl >= min_split) or (cnt - nl >= 2 * min_leaf and cnt - nl >= min_split):
                expand = True
        n_part = p if expand else 1
        for j in range(n_part):
            l = a
            rc = 0
            for k in range(a, b):
                r = ordr[j, k]
                if goes[r] == 1:
                    ordr[j, l] = r
                    l += 1
                else:
                    tmp[rc] = r
                    rc += 1
            for q in range(rc):
                ordr[j, l + q] = tmp[q]
        stack[sp] = rchild
        stack[sp + 1] = lchild
        sp += 2
    return (feat[:n_nodes].copy(), thr[:n_nodes].copy(), left[:n_nodes].copy(), right[:n_nodes].copy(),
            val[:n_nodes].copy(), gain_sum / w_root)


@njit(cache=True, nogil=True)
def predict_rows(feat, thr, left, right, val, X, rows):
    """Class-probability vectors of the given rows of X (serial, may run inside a worker thread)."""
    out = np.empty((rows.shape[0], val.shape[1]))
    for t in range(rows.shape[0]):
        i = rows[t]
        node = 0
        while feat[node] >= 0:
            if X[i, feat[node]] <= thr[node]:
                node = left[node]
            else:
                node = right[node]
        out[t] = val[node]
    return out


@njit(cache=True, parallel=True)
def predict_all(feat, thr, left, right, val, X):
    """Class-probability vectors of all rows of X (parallel over rows)."""
    out = np.empty((X.shape[0], val.shape[1]))
    for i in prange(X.shape[0]):
        node = 0
        while feat[node] >= 0:
            if X[i, feat[node]] <= thr[node]:
                node = left[node]
            else:
                node = right[node]
        out[i] = val[node]
    return out


class DecisionTree:
    """Single classification tree.

    criterion         'gini', 'entropy', 'gain_ratio' or 'c45'
    max_depth         depth limit (None = unlimited)
    min_samples_leaf  minimum number of distinct training rows in a leaf
    min_samples_split minimum number of distinct training rows needed to split a node
    max_features      features drawn per node (None = all)
    count_constants   whether a constant column uses up one of the draws (see module docstring)
    """

    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1, min_samples_split=2,
                 max_features=None, seed=0, n_classes=10, count_constants=False):
        self.criterion, self.max_depth = criterion, max_depth
        self.min_samples_leaf, self.min_samples_split = min_samples_leaf, min_samples_split
        self.max_features, self.seed, self.n_classes = max_features, seed, n_classes
        self.count_constants = count_constants

    def fit(self, X, y, sample_weight=None, order=None):
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.int64)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        order = presort(X) if order is None else order
        m = X.shape[1] if self.max_features is None else int(self.max_features)
        depth = NO_LIMIT if self.max_depth is None else int(self.max_depth)
        (self.feature, self.threshold, self.left, self.right, self.value,
         imp) = _grow(X, y, w, order, self.n_classes, CRITERIA[self.criterion], m,
                                            int(self.min_samples_leaf), int(self.min_samples_split), depth,
                                            int(self.count_constants), int(self.seed))
        self.feature_importances_ = imp / imp.sum() if imp.sum() > 0 else imp
        return self

    def predict_proba(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        return predict_all(self.feature, self.threshold, self.left, self.right, self.value, X)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    @property
    def n_leaves(self):
        return int((self.feature < 0).sum())

    @property
    def depth(self):
        d = np.zeros(len(self.feature), int)
        for i in range(len(self.feature)):
            if self.feature[i] >= 0:
                d[self.left[i]] = d[self.right[i]] = d[i] + 1
        return int(d.max())
