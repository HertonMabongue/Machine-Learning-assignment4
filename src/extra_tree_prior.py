"""Fairness check: give the single tree the same post-hoc class-prior correction as the forest.
Correction weights come from 3-fold cross-validated probabilities on D; G is used only for scoring."""
import sys, json
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RES
import numpy as np
from data import load_clean
from features import split_dev_gen, encode
from ensemble import fit_prior_weights
from tree import DecisionTree, presort
from metrics import summary, per_class
from baselines_common import strat_folds
X, y = load_clean(); D, G = split_dev_gen(y)
yD, yG = y[D], y[G]
cfg = json.load(open(RES + "tree_tuning_full.json"))[0]
Xa, _ = encode(X.iloc[D], X.iloc[D], cfg["encoding"]); Xb, _ = encode(X.iloc[D], X.iloc[G], cfg["encoding"])
Xa, Xb = Xa.astype(np.float32), Xb.astype(np.float32)
order = presort(Xa)
fold = strat_folds(yD, 3, 3)
out = {}
for name, balanced in [("balanced", True), ("unweighted", False)]:
    def weights(mask):
        cnt = np.bincount(yD[mask], minlength=10).astype(float)
        w = len(yD[mask]) / (10.0 * np.maximum(cnt, 1))[yD] if balanced else np.ones(len(yD))
        return w * mask
    mk = lambda: DecisionTree(cfg["criterion"], cfg["max_depth"], 1, cfg["min_samples_split"], None)
    P = np.zeros((len(yD), 10))
    for f in range(3):
        t = mk().fit(Xa, yD, weights(fold != f), order)
        P[fold == f] = t.predict_proba(Xa[fold == f])
    w, _ = fit_prior_weights(P, yD)
    t = mk().fit(Xa, yD, weights(np.ones(len(yD), bool)), order)
    PG = t.predict_proba(Xb)
    out[name] = dict(raw=summary(yG, PG.argmax(1)), prior=summary(yG, (PG * w).argmax(1)), w=w.tolist(),
                     pc_prior=per_class(yG, (PG * w).argmax(1))[2].tolist())
    np.save(RES + "pred_tree_prior_%s.npy" % name, (PG * w).argmax(1))
    print(name, out[name]["raw"], out[name]["prior"], flush=True)
json.dump(out, open(RES + "tree_prior.json", "w"), indent=1)
