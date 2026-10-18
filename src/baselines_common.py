"""Helpers shared by the baseline tuning scripts."""
import numpy as np


def strat_folds(y, k, seed):
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(y), dtype=int)
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        fold[idx] = np.arange(len(idx)) % k
    return fold
