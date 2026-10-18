"""Revised assignment 2 baselines, tuned on the whole development set D.

Tree : own CART (tree.py). Assignment 2 grid (depth limit, class weights, minimum split size) extended
       with the split criterion (Gini, entropy, gain ratio, C4.5 average-gain filter) and the encoding
       of the nominal features. Three-fold stratified cross-validation on all of D, macro-averaged F1.
KNN  : own exact search (knn.py). Stratified 75/25 split of D, exact search over all 75% training rows.
       Distances: Euclidean and Manhattan on the assignment 2 encoding, HEOM and Gower on mixed-type
       data with two numeric scalings. Selection score: cross-fitted prior-corrected macro F1.
"""
import sys, time, json, os
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from joblib import Parallel, delayed
from data import load_clean
from features import split_dev_gen, encode, encode_knn, encode_mixed
from ensemble import fit_prior_weights
from knn import knn_search, votes
from tree import DecisionTree, presort
from metrics import macro_f1, accuracy
from baselines_common import strat_folds

from paths import RES
X, y = load_clean()
D, G = split_dev_gen(y)
XD, yD = X.iloc[D], y[D]


def balanced_weights(y_train_labels, y_all):
    """Per-row weights n / (K n_c) computed on the training rows, as in the 'balanced' option of assignment 2."""
    cnt = np.bincount(y_train_labels, minlength=10).astype(float)
    return len(y_train_labels) / (10.0 * np.maximum(cnt, 1))[y_all]


def tree_cv_score(Xe, order, crit, depth, balanced, min_split, fold):
    sc = []
    for f in range(3):
        tr = fold != f
        w = balanced_weights(yD[tr], yD) if balanced else np.ones(len(yD))
        w = w * tr                                  # held-out rows get weight zero and are ignored
        t = DecisionTree(crit, depth, 1, min_split, None).fit(Xe, yD, w, order)
        sc.append(macro_f1(yD[~tr], t.predict(Xe[~tr])))
    return float(np.mean(sc)), float(np.std(sc))


def done(name, key):
    """True if a result file of the current version exists (old versions lack the key)."""
    return os.path.exists(RES + name) and key in json.dumps(json.load(open(RES + name)))[:400]


if not done("tree_tuning_full.json", "encoding"):
    t0 = time.time()
    fold = strat_folds(yD, 3, 3)
    tree_res = []
    for how in ["onehot", "top", "freq"]:
        Xe = encode(XD, XD, how)[0].astype(np.float32)
        order = presort(Xe)
        grid = [(c, d, b, m) for c in ["gini", "entropy", "gain_ratio", "c45"] for d in [5, 10, 15, 20, None]
                for b in [False, True] for m in [2, 10]]
        out = Parallel(n_jobs=2, backend="threading")(
            delayed(tree_cv_score)(Xe, order, c, d, b, m, fold) for c, d, b, m in grid)
        for (c, d, b, m), (f1, sd) in zip(grid, out):
            rec = dict(encoding=how, criterion=c, max_depth=d, class_weight="balanced" if b else None,
                       min_samples_split=m, f1=f1, sd=sd)
            tree_res.append(rec)
        print("tree encoding", how, "done", round(time.time() - t0), "s; best so far",
              max(tree_res, key=lambda r: r["f1"]), flush=True)
    tree_res.sort(key=lambda r: -r["f1"])
    json.dump(tree_res, open(RES + "tree_tuning_full.json", "w"), indent=1)
    print("BEST TREE", tree_res[0], flush=True)

# KNN section
if done("knn_best_config.json", "scaling"):
    print("KNN tuning already done")
    sys.exit(0)
rng = np.random.default_rng(7)
fold = strat_folds(yD, 4, 7)                        # one quarter held out for validation
tr, va = np.flatnonzero(fold != 0), np.flatnonzero(fold == 0)
va = rng.choice(va, 15000, replace=False)
half = rng.random(len(va)) < 0.5
variants = [("euclidean", None), ("manhattan", None), ("heom", "range"), ("heom", "quantile"),
            ("gower", "range"), ("gower", "quantile")]
knn_res, store = [], {}
A1, names1, _ = encode_knn(XD.iloc[tr], XD.iloc[tr])
B1, _, _ = encode_knn(XD.iloc[tr], XD.iloc[va])
print("KNN one-hot features", A1.shape, flush=True)
for metric, scaling in variants:
    t0 = time.time()
    if scaling is None:
        idx, dist = knn_search(B1, A1, 101, metric)
    else:
        An, Ac, inv = encode_mixed(XD.iloc[tr], XD.iloc[tr], scaling)
        Bn, Bc, _ = encode_mixed(XD.iloc[tr], XD.iloc[va], scaling)
        idx, dist = knn_search(Bn, An, 101, metric, Bc, Ac, inv, clip=True)
    print(metric, scaling, "search secs", round(time.time() - t0), flush=True)
    for k in [3, 5, 7, 11, 15, 21, 31, 51, 101]:
        for wt in ["uniform", "distance"]:
            P = votes(idx, dist, yD[tr], k, wt)
            raw = macro_f1(yD[va], P.argmax(1))
            cf = []
            for a, b in [(half, ~half), (~half, half)]:
                w, _ = fit_prior_weights(P[a], yD[va][a])
                cf.append(macro_f1(yD[va][b], (P[b] * w).argmax(1)))
            rec = dict(metric=metric, scaling=scaling, k=k, weighting=wt, f1=float(raw),
                       acc=accuracy(yD[va], P.argmax(1)), f1_prior=float(np.mean(cf)))
            knn_res.append(rec)
            store[(metric, scaling, k, wt)] = P
    print("best so far", max(knn_res, key=lambda r: r["f1_prior"]), flush=True)
knn_res.sort(key=lambda r: -r["f1_prior"])
json.dump(knn_res, open(RES + "knn_tuning_full.json", "w"), indent=1)
best = knn_res[0]
Pb = store[(best["metric"], best["scaling"], best["k"], best["weighting"])]
w, f = fit_prior_weights(Pb, yD[va])                 # weights for the final model: all validation rows
best["prior_weights"] = w.tolist()
json.dump(best, open(RES + "knn_best_config.json", "w"), indent=1)
print("BEST KNN", best)
