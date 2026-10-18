"""Checks of the own tree: (1) root split against brute force for every criterion,
(2) agreement with the scikit-learn tree on real data at matching settings."""
import sys, time
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tree import DecisionTree, presort

def impurity(Wc, crit):
    W = Wc.sum(); p = Wc / W
    if crit == "gini": return 1 - (p ** 2).sum()
    q = p[p > 0]; return -(q * np.log(q)).sum()

def brute_root(X, y, w, crit, leaf):
    n, p = X.shape; C = 10
    Wc = np.bincount(y, weights=w, minlength=C); W = Wc.sum()
    base = "gini" if crit == "gini" else "entropy"
    imp = impurity(Wc, base)
    cand = []
    for j in range(p):
        vals = np.unique(X[:, j])
        best = (0, None, None)
        for a, b in zip(vals[:-1], vals[1:]):
            t = 0.5 * (float(a) + float(b))
            L = X[:, j] <= t
            if L.sum() < leaf or (~L).sum() < leaf: continue
            wl = np.bincount(y[L], weights=w[L], minlength=C); wr = Wc - wl
            g = imp - wl.sum() / W * impurity(wl, base) - wr.sum() / W * impurity(wr, base)
            if g > best[0] + 1e-12: best = (g, t, wl.sum() / W)
        if best[1] is not None:
            pl = best[2]; si = -(pl * np.log(pl) + (1 - pl) * np.log(1 - pl))
            cand.append((j, best[0], si, best[1]))
    if crit in ("gini", "entropy"):
        return max(cand, key=lambda c: c[1])
    if crit == "c45":
        avg = np.mean([c[1] for c in cand]); cand = [c for c in cand if c[1] >= avg - 1e-12]
    return max(cand, key=lambda c: c[1] / c[2])

rng = np.random.default_rng(1)
ok = True
for trial in range(30):
    n = int(rng.integers(40, 400)); p = int(rng.integers(2, 8))
    X = np.round(rng.normal(size=(n, p)) * rng.integers(1, 4, size=p), int(rng.integers(0, 2))).astype(np.float32)
    y = rng.integers(0, int(rng.integers(2, 6)), size=n)
    w = rng.uniform(0.2, 3.0, size=n)
    leaf = int(rng.integers(1, 6))
    for crit in ["gini", "entropy", "gain_ratio", "c45"]:
        t = DecisionTree(crit, max_depth=1, min_samples_leaf=leaf, max_features=p, seed=0).fit(X, y, w)
        ref = brute_root(X, y, w, crit, leaf)
        got = (int(t.feature[0]), float(t.threshold[0])) if t.feature[0] >= 0 else None
        if got is None or got[0] != ref[0] or abs(got[1] - ref[3]) > 1e-6:
            ok = False; print("MISMATCH", trial, crit, got, ref)
print("brute-force root split check:", "PASS" if ok else "FAIL")

# agreement with scikit-learn on the real data
from sklearn.tree import DecisionTreeClassifier
from data import load_clean
from features import split_dev_gen, encode
X, y = load_clean(); D, G = split_dev_gen(y)
Xd, _ = encode(X.iloc[D], X.iloc[D], "onehot"); Xg, _ = encode(X.iloc[D], X.iloc[G], "onehot")
yd, yg = y[D], y[G]
for crit in ["gini", "entropy"]:
    for depth, leaf in [(5, 1), (15, 1), (None, 1), (None, 10)]:
        t0 = time.time(); mine = DecisionTree(crit, depth, leaf, 2, None).fit(Xd, yd); t1 = time.time() - t0
        sk = DecisionTreeClassifier(criterion=crit, max_depth=depth, min_samples_leaf=leaf, random_state=0).fit(Xd, yd)
        pm, ps = mine.predict(Xg), sk.predict(Xg)
        imp_diff = np.abs(mine.feature_importances_ - sk.feature_importances_).max()
        print(f"{crit:8s} depth={depth} leaf={leaf}: agreement {np.mean(pm == ps):.4f}  acc own {np.mean(pm == yg):.4f} sk {np.mean(ps == yg):.4f}"
              f"  leaves {mine.n_leaves}/{sk.get_n_leaves()}  max|importance diff| {imp_diff:.4f}  fit {t1:.1f}s", flush=True)
