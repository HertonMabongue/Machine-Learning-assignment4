"""Q1: number of trees. One 400-member forest with the final settings; OOB scores are read at checkpoints."""
import sys, json
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RES
import numpy as np
from data import load_clean
from features import split_dev_gen, encode
from ensemble import BaggedForest
X, y = load_clean(); D, G = split_dev_gen(y)
best = json.load(open(RES + "ensemble_best_config.json"))
Xe, names = encode(X.iloc[D], X.iloc[D], best["enc"])
mf = "sqrt" if best["max_features"] == "sqrt" else max(1, int(round(best["max_features"] * len(names))))
f = BaggedForest(n_trees=400, max_features=mf, min_samples_leaf=best["leaf"], max_depth=best["depth"],
                 criterion=best["crit"], alpha=best["alpha"], beta=best["beta"], seed=77).fit(
    Xe, y[D], checkpoints=[1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 400], keep_trees=False)
curve = [dict(trees=t, f1=a, acc=b) for t, a, b in f.curve]
plateau = curve[-1]["f1"]
chosen = next(c["trees"] for c in curve if c["trees"] >= 100 and c["f1"] >= plateau - 0.002)
json.dump(dict(curve=curve, chosen_trees=chosen, member_f1=float(np.mean(f.member_f1)),
               member_f1_sd=float(np.std(f.member_f1))), open(RES + "q1_curve.json", "w"), indent=1)
print(curve, chosen, np.mean(f.member_f1))
