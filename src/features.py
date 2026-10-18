"""Feature encodings for tree ensembles and the data split used by all experiments."""
import numpy as np
from data import load_clean, NOMINAL

SEED_SPLIT = 20261009


def split_dev_gen(y, frac=0.2, seed=SEED_SPLIT):
    """Stratified split into development set D and generalisation set G."""
    rng = np.random.default_rng(seed)
    g = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        g[rng.choice(idx, size=int(round(frac * len(idx))), replace=False)] = True
    return np.flatnonzero(~g), np.flatnonzero(g)


def encode(X_fit, X_apply, how="onehot", proto_top=15):
    """Encode nominal columns. Statistics come from X_fit only.
    onehot : every category of proto, state and service one indicator (assignment 2 encoding)
    top    : as onehot, but proto limited to the proto_top most frequent categories + 'other'
    freq   : one integer column per nominal feature, categories ranked by training frequency
    """
    num = [c for c in X_fit.columns if c not in NOMINAL]
    out = [X_apply[num].to_numpy(dtype=np.float64)]
    names = list(num)
    for c in NOMINAL:
        vc = X_fit[c].value_counts()
        levels = vc.index.tolist()
        col = X_apply[c].to_numpy()
        if how == "freq":
            rank = {v: i for i, v in enumerate(levels)}
            out.append(np.array([rank.get(v, len(levels)) for v in col], dtype=np.float64)[:, None])
            names.append(c + "_rank")
            continue
        if how == "top" and c == "proto":
            levels = levels[:proto_top]
            col = np.where(np.isin(col, levels), col, "other")
            levels = levels + ["other"]
        out.append((col[:, None] == np.array(levels)[None, :]).astype(np.float64))
        names += [f"{c}={v}" for v in levels]
    return np.hstack(out), names


def encode_knn(X_fit, X_apply, proto_top=15, skew_thr=2.0):
    """Assignment 2 pre-processing for distance-based methods, fitted on X_fit only:
    proto limited to its most frequent categories, one-hot nominals, log(1+x) on numeric
    features with |skewness| > skew_thr, then centring on the median and division by the
    interquartile range (unchanged when the range is zero)."""
    num = [c for c in X_fit.columns if c not in NOMINAL]
    A = X_fit[num].astype(float)
    skew = A.skew()
    logc = skew[skew.abs() > skew_thr].index.tolist()
    B = X_apply[num].astype(float)
    for c in logc:
        A[c], B[c] = np.log1p(A[c]), np.log1p(B[c])
    med = A.median()
    iqr = (A.quantile(0.75) - A.quantile(0.25)).replace(0, 1.0)
    parts = [((B - med) / iqr).to_numpy()]
    names = list(num)
    for c in NOMINAL:
        levels = X_fit[c].value_counts().index.tolist()
        col = X_apply[c].to_numpy()
        if c == "proto":
            levels = levels[:proto_top]
            col = np.where(np.isin(col, levels), col, "other")
            levels = levels + ["other"]
        parts.append((col[:, None] == np.array(levels)[None, :]).astype(float))
        names += [f"{c}={v}" for v in levels]
    return np.hstack(parts), names, logc


def encode_mixed(X_fit, X_apply, scaling="range", skew_thr=2.0):
    """Mixed-type representation for HEOM and Gower distances, fitted on X_fit only.
    Numeric part : log(1+x) on features with |skewness| > skew_thr (as in assignment 2), then every
                   feature is divided by its range on X_fit. 'range' uses max - min, 'quantile' uses the
                   range between the 1st and 99th percentile, which an outlier cannot inflate.
    Nominal part : integer category codes (proto, state, service). Unseen categories receive a new code.
    Returns numeric matrix, code matrix and 1/range of every numeric feature."""
    num = [c for c in X_fit.columns if c not in NOMINAL]
    A = X_fit[num].astype(float)
    B = X_apply[num].astype(float)
    skew = A.skew()
    for c in skew[skew.abs() > skew_thr].index:
        A[c], B[c] = np.log1p(A[c]), np.log1p(B[c])
    if scaling == "range":
        rng_ = (A.max() - A.min()).to_numpy()
    else:
        rng_ = (A.quantile(0.99) - A.quantile(0.01)).to_numpy()
    inv = np.divide(1.0, rng_, out=np.zeros_like(rng_), where=rng_ > 0)
    codes = []
    for c in NOMINAL:
        lev = {v: i for i, v in enumerate(X_fit[c].value_counts().index)}
        codes.append(np.array([lev.get(v, len(lev)) for v in X_apply[c].to_numpy()], dtype=np.int32))
    return B.to_numpy(), np.stack(codes, axis=1), inv
