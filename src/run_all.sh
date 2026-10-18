#!/bin/bash
# Full pipeline. Run from anywhere. Needs data/networkTraffic.csv (or the environment variable NB15_CSV).
# Results are deterministic for fixed seeds. 
set -e
cd "$(dirname "$0")"
mkdir -p ../results
python3 tune_ensemble.py   > ../results/tune_ensemble.log 2>&1   # forest tuning, questions Q2-Q7
python3 q1_curve.py        > ../results/q1.log 2>&1               # number of trees, question Q1
python3 baselines.py       > ../results/baselines.log 2>&1        # tree and nearest-neighbour tuning
python3 baselines_boost.py > ../results/baselines_boost.log 2>&1  # boosted ensemble tuning, stage 1
python3 baselines_boost2.py > ../results/baselines_boost2.log 2>&1 # boosted ensemble tuning, stage 2 (learning rate, up to 800 rounds)
python3 final_eval.py 30   > ../results/final_eval.log 2>&1       # generalisation set, ladder, 30 forest runs
python3 extra_tree_prior.py > ../results/extra_tree_prior.log 2>&1 # tree with class-prior correction
python3 stats.py           > ../results/stats.log 2>&1            # bootstrap and McNemar tests
