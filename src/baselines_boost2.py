"""Second stage of the boosting tuning: learning rate and number of rounds (up to 800) at the best setting of stage 1.
Stage 1 (baselines_boost.py) found the best score at the edge of its grid (200 rounds, learning rate 0.25); this script extends the grid.
Tuning of the boosted ensemble (SAMME) for the comparison with the forest.
Same validation protocol as the nearest-neighbour rule: stratified 75/25 split of the development set D,
15,000 validation rows, cross-fitted prior-corrected macro F1 as selection score.
Stage 1: depth limit x balanced initial weights at learning rate 1. Stage 2: learning rate at the best of stage 1.
The number of rounds is read at checkpoints within every fit."""
import sys, time, json
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from data import load_clean
from features import split_dev_gen, encode
from ensemble import fit_prior_weights
from boost import AdaBoost
from tree import presort
from metrics import macro_f1, accuracy
from baselines_common import strat_folds

from paths import RES
X, y = load_clean(); D, G = split_dev_gen(y)
XD, yD = X.iloc[D], y[D]
enc = json.load(open(RES + "ensemble_best_config.json"))["enc"]
Xe = encode(XD, XD, enc)[0].astype(np.float32)
order = presort(Xe)
rng = np.random.default_rng(7)
fold = strat_folds(yD, 4, 7)                       # the split of baselines.py (same seed, same draws)
tr_mask = fold != 0
va = np.flatnonzero(fold == 0)
va = rng.choice(va, 15000, replace=False)
half = rng.random(len(va)) < 0.5
CHECK = [25, 50, 100, 150, 200, 300, 400, 600, 800]
results, store = [], {}

def run(depth, balanced, lr):
    t0 = time.time()
    m = AdaBoost(max(CHECK), depth, lr, balanced).fit(Xe, yD, order, tr_mask)
    S = m.staged_scores(Xe[va], CHECK)
    for r, P in S.items():
        cf = []
        for a, b in [(half, ~half), (~half, half)]:
            w, _ = fit_prior_weights(P[a], yD[va][a])
            cf.append(macro_f1(yD[va][b], (P[b] * w).argmax(1)))
        rec = dict(depth=depth, balanced=balanced, lr=lr, rounds=r, f1=float(macro_f1(yD[va], P.argmax(1))),
                   acc=accuracy(yD[va], P.argmax(1)), f1_prior=float(np.mean(cf)), members=len(m.trees))
        results.append(rec); store[(depth, balanced, lr, r)] = P
    print(depth, balanced, lr, "secs", round(time.time() - t0), "members", len(m.trees),
          [round(x["f1_prior"], 4) for x in results[-len(S):]], flush=True)

prev = json.load(open(RES + "boost_tuning.json"))
s1 = [r for r in prev if r["lr"] == 1.0]                      # stage 1 results stay
b = max(s1, key=lambda r: r["f1_prior"])
print("stage 1 best", b["depth"], b["balanced"], flush=True)
for lr in [0.5, 0.25, 0.1]:
    run(b["depth"], b["balanced"], lr)
results = s1 + results
results.sort(key=lambda r: -r["f1_prior"])
json.dump(results, open(RES + "boost_tuning.json", "w"), indent=1)
best = results[0]
w, _ = fit_prior_weights(store[(best["depth"], best["balanced"], best["lr"], best["rounds"])], yD[va])
best["prior_weights"] = w.tolist()
best["enc"] = enc
json.dump(best, open(RES + "boost_best_config.json", "w"), indent=1)
print("BEST BOOST", best)
