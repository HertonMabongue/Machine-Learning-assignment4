"""Data loading shared by all experiments (replicates the assignment 2 clean-up)."""
import numpy as np
import pandas as pd

from paths import find_data
NOMINAL = ["proto", "state", "service"]


def load_clean():
    """Load, deduplicate (features + label, id excluded), drop id. Returns X (DataFrame), y (ndarray)."""
    df = pd.read_csv(find_data(), na_values="?")
    cols = [c for c in df.columns if c != "id"]
    df = df.drop_duplicates(subset=cols, keep="first").reset_index(drop=True)
    df = df.drop(columns=["id"])
    y = df["attack_cat"].astype(int).to_numpy()
    X = df.drop(columns=["attack_cat"])
    X[NOMINAL] = X[NOMINAL].fillna("missing")
    return X, y
