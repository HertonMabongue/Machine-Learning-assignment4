"""
Multi-class AdaBoost (SAMME, Zhu et al., 2009) written from scratch on the own tree.

Round m grows a depth-limited tree on the weighted training set, computes its weighted error e_m
and the member weight a_m = lr * (log((1 - e_m) / e_m) + log(K - 1)), then multiplies the weight of
every misclassified instance by exp(a_m). Boosting stops early when e_m reaches zero or when e_m is
not below 1 - 1/K. The class scores are s_c(x) = sum_m a_m [h_m(x) = c], normalised to a sum of one.
With balanced=True the initial weights are inversely proportional to the class frequency.
"""
import numpy as np
from tree import DecisionTree, presort

N_CLASSES = 10


class AdaBoost:
    def __init__(self, n_rounds=100, max_depth=6, learning_rate=1.0, balanced=True, criterion="gini"):
        self.n_rounds, self.max_depth, self.lr = n_rounds, max_depth, learning_rate
        self.balanced, self.criterion = balanced, criterion

    def fit(self, X, y, order=None, row_mask=None):
        """row_mask selects the training rows (others are ignored), so one presorting serves all folds."""
        X = np.ascontiguousarray(X, dtype=np.float32)
        order = presort(X) if order is None else order
        use = np.ones(len(y), bool) if row_mask is None else row_mask
        cnt = np.bincount(y[use], minlength=N_CLASSES).astype(float)
        w = np.where(use, (1.0 / np.maximum(cnt, 1))[y] if self.balanced else 1.0, 0.0)
        w = w / w.sum()
        self.trees, self.alphas, self.train_err = [], [], []
        scale = use.sum()                                  # weights handed to the tree sum to the sample size
        for m in range(self.n_rounds):
            t = DecisionTree(self.criterion, self.max_depth, 1, 2, None).fit(X, y, w * scale, order)
            wrong = (t.predict(X) != y) & use
            err = w[wrong].sum()
            if err <= 1e-12:
                self.trees.append(t); self.alphas.append(10.0); self.train_err.append(0.0)
                break
            if err >= 1.0 - 1.0 / N_CLASSES:
                break
            a = self.lr * (np.log((1 - err) / err) + np.log(N_CLASSES - 1))
            w = w * np.exp(a * wrong)
            w = w / w.sum()
            self.trees.append(t); self.alphas.append(a); self.train_err.append(float(err))
        return self

    def scores(self, X, rounds=None):
        """Normalised class scores using the first `rounds` members (all by default)."""
        X = np.ascontiguousarray(X, dtype=np.float32)
        S = np.zeros((len(X), N_CLASSES))
        r = len(self.trees) if rounds is None else min(rounds, len(self.trees))
        for t, a in zip(self.trees[:r], self.alphas[:r]):
            S[np.arange(len(X)), t.predict(X)] += a
        tot = S.sum(axis=1, keepdims=True)
        return S / np.where(tot > 0, tot, 1.0)

    def staged_scores(self, X, checkpoints):
        """Scores at several numbers of rounds with a single pass over the members."""
        X = np.ascontiguousarray(X, dtype=np.float32)
        S = np.zeros((len(X), N_CLASSES))
        out = {}
        marks = set(checkpoints)
        for r, (t, a) in enumerate(zip(self.trees, self.alphas), start=1):
            S[np.arange(len(X)), t.predict(X)] += a
            if r in marks:
                tot = S.sum(axis=1, keepdims=True)
                out[r] = S / np.where(tot > 0, tot, 1.0)
        last = len(self.trees)
        if last not in out:
            tot = S.sum(axis=1, keepdims=True)
            out[last] = S / np.where(tot > 0, tot, 1.0)
        return out
