"""Sequential, pre-registered tuning of the forest on the development set using OOB estimates.
Stages answer questions Q2-Q6 of the report (Q1 is answered by q1_curve.py). The best setting of a
stage is carried forward to the next stage."""
import sys, time, json, os
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RES
import numpy as np
from data import load_clean
from features import split_dev_gen, encode
from ensemble import BaggedForest, fit_prior_weights
from tree import presort
from metrics import macro_f1, accuracy

OUT = RES + "ensemble_tuning.jsonl"
N_TREES = 60
SEEDS = [11, 12]

X, y = load_clean()
D, G = split_dev_gen(y)
XD, yD = X.iloc[D], y[D]
enc_cache = {}
def get_enc(how):
    """Encoded matrix, column names and the presorting of every column (computed once per encoding)."""
    if how not in enc_cache:
        M, names = encode(XD, XD, how)
        M = M.astype(np.float32)
        enc_cache[how] = (M, names, presort(M))
    return enc_cache[how]

rng = np.random.default_rng(5)
half = rng.random(len(yD)) < 0.5            # fixed split of OOB rows for cross-fitted prior correction

def crossfit(P, m, yy):
    """Fit class-prior weights on one half of the OOB rows, score on the other, and swap."""
    idx = np.flatnonzero(m)
    a, b = idx[half[idx]], idx[~half[idx]]
    out = []
    for fit, ev in [(a, b), (b, a)]:
        w, _ = fit_prior_weights(P[fit], yy[fit])
        out.append(macro_f1(yy[ev], (P[ev] * w).argmax(1)))
    return float(np.mean(out))

def run(stage, cfg):
    Xe, names, order = get_enc(cfg["enc"])
    p = len(names)
    mf = cfg["max_features"]
    mf = mf if mf == "sqrt" else max(1, int(round(mf * p)))
    res = []
    for s in SEEDS:
        t = time.time()
        f = BaggedForest(n_trees=N_TREES, max_features=mf, min_samples_leaf=cfg["leaf"], max_depth=cfg["depth"],
                         criterion=cfg["crit"], alpha=cfg["alpha"], beta=cfg["beta"], seed=s,
                         count_constants=cfg["cc"]).fit(Xe, yD, order=order)
        P, m = f.oob_proba()
        res.append(dict(f1=macro_f1(yD[m], P[m].argmax(1)), acc=accuracy(yD[m], P[m].argmax(1)),
                        f1_prior=crossfit(P, m, yD), member_f1=float(np.mean(f.member_f1)),
                        secs=time.time() - t))
    rec = dict(stage=stage, cfg=cfg, n_features=p,
               **{k: float(np.mean([r[k] for r in res])) for k in res[0]},
               **{k + "_sd": float(np.std([r[k] for r in res])) for k in ["f1", "acc", "f1_prior"]})
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(stage, cfg, {k: round(v, 4) for k, v in rec.items() if isinstance(v, float)}, flush=True)
    return rec

best = dict(enc="onehot", max_features="sqrt", leaf=1, depth=None, crit="gini", alpha=1.0, beta=0.0, cc=False)
def pick(recs):
    return max(recs, key=lambda r: r["f1_prior"])["cfg"]

if os.path.exists(OUT): os.remove(OUT)
# Q4 class handling
best = pick([run("Q4_class", {**best, "alpha": a, "beta": b}) for a, b in
             [(1, 0), (1, 0.5), (1, 1), (0.5, 0), (0.5, 0.5), (0, 0)]])
# Q5 encoding
before_q5 = best
best = pick([run("Q5_encoding", {**best, "enc": e}) for e in ["onehot", "top", "freq"]])
# informative only: the same comparison with the textbook feature draw (constant columns use up draws)
for e in ["onehot", "top", "freq"]:
    run("Q5x_encoding_textbook_draw", {**before_q5, "enc": e, "cc": True})
# Q2 feature subset size (the grid reaches all features, which equals plain bagging)
best = pick([run("Q2_features", {**best, "max_features": m}) for m in ["sqrt", 0.1, 0.25, 0.5, 0.75, 1.0]])
# Q2c feature draw rule: m varying columns, or m draws that may hit constant columns (textbook, scikit-learn)
best = pick([run("Q2c_draw", {**best, "cc": c}) for c in [False, True]])
# Q3 tree complexity
best = pick([run("Q3_leaf", {**best, "leaf": l}) for l in [1, 2, 5, 10, 20, 50]])
best = pick([run("Q3_depth", {**best, "depth": d}) for d in [10, 15, 20, None]])
# Q6 split criterion
best = pick([run("Q6_criterion", {**best, "crit": c}) for c in ["gini", "entropy", "gain_ratio", "c45"]])
json.dump(best, open(RES + "ensemble_best_config.json", "w"))
print("BEST", best)
