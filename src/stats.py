"""Paired bootstrap and McNemar tests on the generalisation set."""
import sys, json
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import chi2
from metrics import macro_f1, accuracy
from paths import RES as R
y = np.load(R + "yG.npy")
P = {"forest": np.load(R + "pred_forest.npy"), "tree": np.load(R + "pred_tree.npy"), "knn": np.load(R + "pred_knn.npy"),
     "tree_prior": np.load(R + "pred_tree_prior_balanced.npy"), "tree_prior_unweighted": np.load(R + "pred_tree_prior_unweighted.npy"),
     "boost": np.load(R + "pred_boost.npy")}
rng = np.random.default_rng(2026)
B = 1000
idx = rng.integers(0, len(y), size=(B, len(y)))
def boot(pred):
    f = np.empty(B); a = np.empty(B)
    for b in range(B):
        i = idx[b]
        f[b] = macro_f1(y[i], pred[i]); a[b] = (y[i] == pred[i]).mean()
    return f, a
res = {}
bs = {k: boot(v) for k, v in P.items()}
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
for k, (f, a) in bs.items():
    res[k] = dict(macro_f1=float(macro_f1(y, P[k])), macro_f1_ci=ci(f), acc=float(accuracy(y, P[k])), acc_ci=ci(a))
for a, b in [("forest", "tree"), ("forest", "knn"), ("tree", "knn"), ("forest", "tree_prior"), ("forest", "tree_prior_unweighted"), ("tree_prior", "knn"), ("forest", "boost"), ("boost", "tree_prior"), ("boost", "knn")]:
    d = bs[a][0] - bs[b][0]
    da = bs[a][1] - bs[b][1]
    ca, cb = P[a] == y, P[b] == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    res[f"{a}_vs_{b}"] = dict(diff_f1=float(macro_f1(y, P[a]) - macro_f1(y, P[b])), diff_f1_ci=ci(d), frac_pos=float((d > 0).mean()),
                              diff_acc=float(accuracy(y, P[a]) - accuracy(y, P[b])), diff_acc_ci=ci(da),
                              only_first=n01, only_second=n10, mcnemar_chi2=float(stat), mcnemar_p=float(chi2.sf(stat, 1)))
json.dump(res, open(R + "stats.json", "w"), indent=1)
print(json.dumps(res, indent=1))
