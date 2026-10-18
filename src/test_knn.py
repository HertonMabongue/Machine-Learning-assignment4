"""Checks of the own neighbour search against scikit-learn (euclidean, manhattan) and a plain numpy
reference (heom, gower)."""
import sys, time; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from knn import knn_search
from sklearn.neighbors import NearestNeighbors
rng = np.random.default_rng(0)
T = rng.normal(size=(3000, 30)).astype(np.float32); Q = rng.normal(size=(200, 30)).astype(np.float32)
for m in ["euclidean", "manhattan"]:
    i, d = knn_search(Q, T, 7, m)
    d2, i2 = NearestNeighbors(n_neighbors=7, metric=m).fit(T).kneighbors(Q)
    print(m, "max distance diff", float(np.abs(d - d2).max()), "same indices", float((i == i2).mean()))

# mixed data reference
pn, pc = 8, 3
Tn = rng.uniform(0, 5, size=(2000, pn)).astype(np.float32); Qn = rng.uniform(-1, 6, size=(150, pn)).astype(np.float32)
Tc = rng.integers(0, 4, size=(2000, pc)).astype(np.int32); Qc = rng.integers(0, 5, size=(150, pc)).astype(np.int32)
inv = 1.0 / rng.uniform(2, 6, size=pn)
for m in ["heom", "gower"]:
    i, d = knn_search(Qn, Tn, 9, m, Qc, Tc, inv, clip=True)
    dn = np.minimum(1.0, np.abs(Qn[:, None, :].astype(float) - Tn[None, :, :]) * inv)       # (nq, nt, pn)
    dc = (Qc[:, None, :] != Tc[None, :, :]).astype(float)
    full = np.sqrt((dn ** 2).sum(2) + (dc ** 2).sum(2)) if m == "heom" else (dn.sum(2) + dc.sum(2)) / (pn + pc)
    ref_i = np.argsort(full, axis=1, kind="stable")[:, :9]
    ref_d = np.take_along_axis(full, ref_i, 1)
    print(m, "max distance diff", float(np.abs(d - ref_d).max()), "same indices", float((i == ref_i).mean()))
# speed on the real size
Tbig = rng.normal(size=(97000, 45)).astype(np.float32); Qbig = rng.normal(size=(2000, 45)).astype(np.float32)
for m in ["euclidean", "manhattan"]:
    t = time.time(); knn_search(Qbig, Tbig, 101, m); print(m, "2000 queries x 97000 rows:", round(time.time() - t, 1), "s")
