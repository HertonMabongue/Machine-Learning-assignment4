"""Final evaluation on the generalisation set G (this was never used for tuning).
Usage: python3 final_eval.py [number of forest runs] [sections...]
Sections: baselines, boost, ladder, final. Without section names all sections are computed.
Results of sections that are not recomputed are kept from final_results.json."""
import sys, os, time, json
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from data import load_clean
from features import split_dev_gen, encode, encode_knn, encode_mixed
from ensemble import BaggedForest, fit_prior_weights
from knn import knn_search, votes
from tree import DecisionTree, presort
from boost import AdaBoost
from metrics import summary, per_class

from paths import RES
N_FINAL = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SECTIONS = set(sys.argv[2:]) or {"baselines", "boost", "ladder", "final"}
N_LADDER = 30
X, y = load_clean()
D, G = split_dev_gen(y)
XD, XG, yD, yG = X.iloc[D], X.iloc[G], y[D], y[G]
best = json.load(open(RES + "ensemble_best_config.json"))
T_FINAL = json.load(open(RES + "q1_curve.json"))["chosen_trees"]
enc = {}
for h in ["onehot", "top", "freq"]:
    A, names = encode(XD, XD, h)
    B, _ = encode(XD, XG, h)
    A = A.astype(np.float32)
    enc[h] = (A, B.astype(np.float32), presort(A), names)
out = json.load(open(RES + "final_results.json")) if os.path.exists(RES + "final_results.json") and len(sys.argv) > 2 else {}


def mf(cfg, p):
    return "sqrt" if cfg["max_features"] == "sqrt" else max(1, int(round(cfg["max_features"] * p)))


def run_forest(cfg, seed, n_trees, want_importance=False):
    Xa, Xb, order, _ = enc[cfg["enc"]]
    f = BaggedForest(n_trees=n_trees, max_features=mf(cfg, Xa.shape[1]), min_samples_leaf=cfg["leaf"],
                     max_depth=cfg["depth"], criterion=cfg["crit"], alpha=cfg["alpha"], beta=cfg["beta"],
                     seed=seed, count_constants=cfg["cc"]).fit(Xa, yD, order=order)
    P, m = f.oob_proba()
    w, _ = fit_prior_weights(P[m], yD[m])
    PG = f.predict_proba(Xb)
    r = dict(raw=summary(yG, PG.argmax(1)), prior=summary(yG, (PG * w).argmax(1)),
             pc_raw=per_class(yG, PG.argmax(1))[2].tolist(), pc_prior=per_class(yG, (PG * w).argmax(1))[2].tolist(),
             w=w.tolist(), member_f1=float(np.mean(f.member_f1)))
    imp = f.feature_importances(Xa.shape[1]) if want_importance else None
    return r, (PG * w).argmax(1), PG.argmax(1), imp


def fit_tree(cfg, enc_name):
    """Single tree on all of D with a tuning configuration of baselines.py."""
    Xa, Xb, order, _ = enc[enc_name]
    cnt = np.bincount(yD, minlength=10).astype(float)
    w = len(yD) / (10.0 * cnt)[yD] if cfg["class_weight"] == "balanced" else np.ones(len(yD))
    t = DecisionTree(cfg["criterion"], cfg["max_depth"], 1, cfg["min_samples_split"], None).fit(Xa, yD, w, order)
    return t, t.predict(Xb)


# tree and nearest neighbours on G
tcfg = json.load(open(RES + "tree_tuning_full.json"))[0]
if "baselines" in SECTIONS:
    tree, pt = fit_tree(tcfg, tcfg["encoding"])
    out["tree"] = dict(cfg=tcfg, **summary(yG, pt), pc=per_class(yG, pt)[2].tolist(), leaves=tree.n_leaves, depth=tree.depth)
    a2cfg = dict(criterion="entropy", max_depth=15, class_weight="balanced", min_samples_split=10)
    _, pa2 = fit_tree(a2cfg, "onehot")
    out["tree_a2_config"] = dict(cfg=a2cfg, **summary(yG, pa2), pc=per_class(yG, pa2)[2].tolist())
    print("tree", out["tree"]["macro_f1"], "tree A2 config", out["tree_a2_config"]["macro_f1"], flush=True)

    kc = json.load(open(RES + "knn_best_config.json"))
    t0 = time.time()
    if kc["scaling"] is None:
        A, _, _ = encode_knn(XD, XD)
        B, _, _ = encode_knn(XD, XG)
        idx, dist = knn_search(B, A, kc["k"], kc["metric"])
    else:
        An, Ac, inv = encode_mixed(XD, XD, kc["scaling"])
        Bn, Bc, _ = encode_mixed(XD, XG, kc["scaling"])
        idx, dist = knn_search(Bn, An, kc["k"], kc["metric"], Bc, Ac, inv, clip=True)
    PK = votes(idx, dist, yD, kc["k"], kc["weighting"])
    wk = np.array(kc["prior_weights"])
    out["knn"] = dict(cfg={k: v for k, v in kc.items() if k != "prior_weights"}, secs=time.time() - t0,
                      raw=summary(yG, PK.argmax(1)), prior=summary(yG, (PK * wk).argmax(1)),
                      pc_raw=per_class(yG, PK.argmax(1))[2].tolist(), pc_prior=per_class(yG, (PK * wk).argmax(1))[2].tolist())
    pk = (PK * wk).argmax(1)
    print("knn", out["knn"]["raw"], out["knn"]["prior"], flush=True)
    np.save(RES + "pred_tree.npy", pt); np.save(RES + "pred_knn.npy", pk); np.save(RES + "yG.npy", yG)
    json.dump(out, open(RES + "final_results.json", "w"), indent=1)

# boosted ensemble on G
if "boost" in SECTIONS:
    bc = json.load(open(RES + "boost_best_config.json"))
    Xa, Xb, order, _ = enc[bc["enc"]]
    bm = AdaBoost(bc["rounds"], bc["depth"], bc["lr"], bc["balanced"]).fit(Xa, yD, order)
    PB = bm.scores(Xb)
    wb = np.array(bc["prior_weights"])
    pb = (PB * wb).argmax(1)
    out["boost"] = dict(cfg={k: v for k, v in bc.items() if k != "prior_weights"}, members=len(bm.trees),
                        raw=summary(yG, PB.argmax(1)), prior=summary(yG, pb),
                        pc_raw=per_class(yG, PB.argmax(1))[2].tolist(), pc_prior=per_class(yG, pb)[2].tolist())
    print("boost", out["boost"]["raw"], out["boost"]["prior"], flush=True)
    np.save(RES + "pred_boost.npy", pb)
    json.dump(out, open(RES + "final_results.json", "w"), indent=1)

#  ladder (N_LADDER runs each; stages that change nothing are skipped)
if "ladder" in SECTIONS:
    base = dict(enc="onehot", max_features="sqrt", leaf=1, depth=None, crit="gini", alpha=1.0, beta=0.0, cc=True)
    steps = [("plain random forest", {}),
             ("class weights", {k: best[k] for k in ["alpha", "beta"]}),
             ("encoding", {"enc": best["enc"]}),
             ("features per split", {"max_features": best["max_features"]}),
             ("feature draw rule", {"cc": best["cc"]}),
             ("leaf size, depth, criterion", {k: best[k] for k in ["leaf", "depth", "crit"]})]
    out["ladder"] = {k: v for k, v in out.get("ladder", {}).items() if len(v["raw_f1"]) == N_LADDER}   # resume after an interruption
    cfg, prev_name = dict(base), None
    for name, change in steps:
        new = {**cfg, **change}
        # the selected setting equals the previous one
        if prev_name is not None and new == cfg:           
            out["ladder"][name] = out["ladder"][prev_name]
            continue
        cfg = new
        if name in out["ladder"] and out["ladder"][name]["cfg"] == cfg:
            prev_name = name
            continue
        rs = [run_forest(cfg, 100 + s, T_FINAL)[0] for s in range(N_LADDER)]
        out["ladder"][name] = dict(cfg=dict(cfg), raw_f1=[r["raw"]["macro_f1"] for r in rs], prior_f1=[r["prior"]["macro_f1"] for r in rs],
                                  raw_acc=[r["raw"]["accuracy"] for r in rs], prior_acc=[r["prior"]["accuracy"] for r in rs])
        prev_name = name
        print(name, np.mean(out["ladder"][name]["raw_f1"]), np.mean(out["ladder"][name]["prior_f1"]), flush=True)
        json.dump(out, open(RES + "final_results.json", "w"), indent=1)

#  final runs
if "final" in SECTIONS:
    runs, preds = [], []
    imp_acc = None
    for s in range(N_FINAL):
        r, pp, pr, imp = run_forest(best, 1000 + s, T_FINAL, want_importance=(s == 0))
        if s == 0:
            imp_acc = imp
        runs.append(r); preds.append(pp)
        print("final run", s, r["prior"], flush=True)
    names = enc[best["enc"]][3]
    out["final_runs"] = runs
    out["final_trees"] = T_FINAL
    order_ = np.argsort([r["prior"]["macro_f1"] for r in runs])
    med = int(order_[len(order_) // 2])
    np.save(RES + "pred_forest.npy", preds[med])
    np.save(RES + "pred_forest_all.npy", np.array(preds, dtype=np.int8))
    imp_order = np.argsort(-imp_acc)[:15]
    out["importance"] = [(names[i], float(imp_acc[i])) for i in imp_order]
    tree, _ = fit_tree(tcfg, tcfg["encoding"])
    out["tree_importance"] = [(enc[tcfg["encoding"]][3][i], float(tree.feature_importances_[i]))
                              for i in np.argsort(-tree.feature_importances_)[:15]]
    json.dump(out, open(RES + "final_results.json", "w"), indent=1)
print("done")
