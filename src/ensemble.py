"""
Bagged classification-tree ensemble (random forest family) written from scratch:
class-aware bootstrap sampling, class weighting, random feature subsets, out-of-bag (OOB)
probability estimates, probability voting and post-hoc class-prior correction.

The base learner is the own CART implementation of tree.py, so every part of the ensemble,
including the split search, is implemented here. Trees are grown in parallel threads. The
compiled tree code releases the interpreter lock, so threads share the data without copies.
"""
import os
import numpy as np
from joblib import Parallel, delayed

from metrics import macro_f1
from tree import CRITERIA, NO_LIMIT, _grow, predict_rows, predict_all, presort

N_CLASSES = 10


def _draw_counts(y, n_draw_per_class, rng):
    """Bootstrap multiplicities: draw n_draw_per_class[c] instances with replacement from class c."""
    counts = np.zeros(len(y), dtype=np.int64)
    for c in range(N_CLASSES):
        idx = np.flatnonzero(y == c)
        m = n_draw_per_class[c]
        counts += np.bincount(rng.choice(idx, size=m, replace=True), minlength=len(y))
    return counts


def _fit_one(X, y, order, seed, params, draws, cw, keep_tree):
    rng = np.random.default_rng(seed)
    counts = _draw_counts(y, draws, rng)
    w = counts * cw[y]                                           # bootstrap multiplicity times class weight
    oob = np.flatnonzero(counts == 0)
    depth = NO_LIMIT if params["max_depth"] is None else int(params["max_depth"])
    tree = _grow(X, y, w, order, N_CLASSES, CRITERIA[params["criterion"]], int(params["max_features"]),
                 int(params["min_samples_leaf"]), 2, depth, int(params["count_constants"]),
                 int(rng.integers(2 ** 31 - 1)))
    proba = predict_rows(tree[0], tree[1], tree[2], tree[3], tree[4], X, oob).astype(np.float32)
    single = macro_f1(y[oob], proba.argmax(axis=1))              # strength of this member alone
    return (tree if keep_tree else None), oob, proba, single


class BaggedForest:
    """
    n_trees          number of ensemble members
    max_features     features examined per split: int, float fraction, 'sqrt', or None (all)
    min_samples_leaf minimum distinct training rows per leaf
    max_depth        depth limit (None = fully grown)
    criterion        'gini', 'entropy', 'gain_ratio' or 'c45' (see tree.py)
    count_constants  False: m columns that vary in the node are examined; True: constant columns use up draws
    alpha            sampling exponent: draws per class proportional to N_c ** alpha
                     (1 = ordinary bootstrap, 0 = equal draws per class); total draws = N
    beta             class-weight exponent: weight of class c proportional to N_c ** (-beta)
                     (0 = none, 1 = fully balanced)
    """
    def __init__(self, n_trees=100, max_features="sqrt", min_samples_leaf=1, max_depth=None,
                 criterion="gini", alpha=1.0, beta=0.0, seed=0, n_jobs=-1, count_constants=False):
        self.n_trees, self.seed, self.n_jobs = n_trees, seed, n_jobs
        self.params = dict(max_features=max_features, min_samples_leaf=min_samples_leaf,
                           max_depth=max_depth, criterion=criterion, count_constants=count_constants)
        self.alpha, self.beta = alpha, beta
        self.trees = []

    def fit(self, X, y, checkpoints=(), keep_trees=True, order=None):
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.int64)
        order = presort(X) if order is None else order           # sorting of every feature, done once
        mf = self.params["max_features"]
        mf = X.shape[1] if mf is None else max(1, int(np.sqrt(X.shape[1]))) if mf == "sqrt" else int(mf)
        params = dict(self.params, max_features=mf)
        self._y = y
        n = len(y)
        totals = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
        share = totals ** self.alpha
        draws = np.maximum(1, np.round(n * share / share.sum())).astype(np.int64)
        cw = totals ** (-self.beta)
        cw = cw / (cw * totals).sum() * n                      # mean weight of one instance = 1
        seeds = np.random.SeedSequence(self.seed).generate_state(self.n_trees)
        # members are fitted in batches so that the OOB curve can be read at checkpoints
        self.oob_sum = np.zeros((n, N_CLASSES))
        self.oob_cnt = np.zeros(n)
        # (n_trees, OOB macro F1, OOB accuracy)
        self.curve = []                                         
        self.trees = []
        self.member_f1 = []                                     
        # macro F1 of each member on its own OOB rows
        done = 0
        self.n_jobs = os.cpu_count() if self.n_jobs in (-1, None) else self.n_jobs
        batch = max(self.n_jobs * 2, 8)
        marks = sorted(set(int(c) for c in checkpoints if c <= self.n_trees) | {self.n_trees})
        while done < self.n_trees:
            # batches end exactly at checkpoints
            step = min(batch, min(m for m in marks if m > done) - done)   
            out = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(_fit_one)(X, y, order, seeds[done + i], params, draws, cw, keep_trees)
                for i in range(step))
            for tree, oob, proba, single in out:
                self.member_f1.append(single)
                self.oob_sum[oob] += proba
                self.oob_cnt[oob] += 1
                if keep_trees:
                    self.trees.append(tree)
            done += step
            if any(done >= m and (done - step) < m for m in marks):
                p, mk = self.oob_proba()
                self.curve.append((done, self._y_f1(p, mk, None), float((self._y[mk] == p[mk].argmax(1)).mean())))
        return self

    # Out of Bag
    def oob_mask(self):
        return self.oob_cnt > 0

    def oob_proba(self):
        m = self.oob_mask()
        p = np.zeros_like(self.oob_sum)
        p[m] = self.oob_sum[m] / self.oob_cnt[m][:, None]
        return p, m

    def oob_macro_f1(self, w=None):
        p, m = self.oob_proba()
        return self._y_f1(p, m, w)

    def _y_f1(self, p, m, w):
        pred = (p[m] * (1.0 if w is None else w)).argmax(axis=1)
        return macro_f1(self._y[m], pred)

    #  predict
    def predict_proba(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        total = np.zeros((len(X), N_CLASSES))
        for t in self.trees:
            total += predict_all(t[0], t[1], t[2], t[3], t[4], X)
        return total / len(self.trees)

    def feature_importances(self, n_features):
        """Mean decrease in impurity of the trees, normalised to a sum of one per tree and then averaged."""
        imp = np.zeros(n_features)
        for t in self.trees:
            v = t[5]
            imp += v / v.sum() if v.sum() > 0 else v
        return imp / len(self.trees)

    def predict(self, X, w=None):
        p = self.predict_proba(X)
        return (p * (1.0 if w is None else w)).argmax(axis=1)


def fit_prior_weights(P, y, rounds=3, grid=None):
    """Post-hoc class-prior correction. Coordinate ascent on macro F1 of argmax(P * w).
    P and y must come from data not used to fit the members (OOB or validation rows)."""
    grid = np.array([0.25, 0.35, 0.5, 0.7, 0.85, 1.0, 1.2, 1.4, 2.0, 2.8, 4.0]) if grid is None else grid
    w = np.ones(N_CLASSES)
    best = macro_f1(y, (P * w).argmax(axis=1))
    for _ in range(rounds):
        for c in range(N_CLASSES):
            keep = w[c]
            for g in grid:
                w2 = w.copy()
                w2[c] = keep * g
                s = macro_f1(y, (P * w2).argmax(axis=1))
                if s > best + 1e-9:
                    best, w = s, w2
    return w / w.sum() * N_CLASSES, best
