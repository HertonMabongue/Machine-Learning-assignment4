# Ensembles on UNSW-NB15

A class-aware random forest, with a boosted ensemble and a k-nearest-neighbour rule as comparison models.

Assignment 2 code: https://github.com/HertonMabongue/Machine-Learning-assignment2

## What is done

- **From scratch** : CART tree with exact splits, sample weights and four criteria
  (Gini, entropy, gain ratio, C4.5 filter) in `src/tree.py`; random forest with out-of-bag predictions, sampling and
  weight exponents and class-prior correction in `src/ensemble.py`; boosting in `src/boost.py`; exact KNN with
  Euclidean, Manhattan, HEOM and Gower distances in `src/knn.py`.
- **Evaluation**: duplicates removed before an 80/20 stratified split, tuning on the development part only, 30 forest runs
  on the held-out part, paired bootstrap and McNemar tests, and a ladder that adds one design decision at a time.

## Run

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd src && python3 test_tree.py && python3 test_knn.py   # unit tests (numba compiles on first use)
bash run_all.sh                                          # full pipeline
```

- **Data**: The plain `data/networkTraffic.csv`
- `python3 final_eval.py 30 ladder` recomputes one section only (`baselines`, `boost`, `ladder`, `final`).
- `results/` holds the saved outputs behind every number in the report.

## Contact

**Herton Mabongue**
27190552@sun.ac.za
