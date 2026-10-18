"""File locations shared by all scripts, independent of the working directory.

The project root is the folder that holds src/, results/ and report/. The data file is searched in this order:
  1. the environment variable NB15_CSV (a path to networkTraffic.csv or networkTraffic.csv.gz),
  2. <root>/data/networkTraffic.csv 
"""
import os

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
RES = os.path.join(ROOT, "results") + os.sep
os.makedirs(RES, exist_ok=True)


def find_data():
    candidates = [os.environ.get("NB15_CSV", "")]
    parent = os.path.dirname(ROOT)
    for name in ("networkTraffic.csv", "networkTraffic.csv.gz"):
        candidates.append(os.path.join(ROOT, "data", name))
        for sibling in ("27190552_ML_assignment2", "Machine-Learning-assignment2"):
            candidates.append(os.path.join(parent, sibling, "data", name))
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError("networkTraffic.csv not found. Put it in <project>/data/ or set NB15_CSV to its path. Searched: "
                            + ", ".join(c for c in candidates if c))
