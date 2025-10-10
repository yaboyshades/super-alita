from __future__ import annotations

import glob

import pandas as pd


def prepare_ml_dataset(data_pattern: str, label_column: str = "target"):
    """Load & split ML dataset assembled from multiple CSVs."""
    files = glob.glob(data_pattern)
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    X = df.drop(columns=[label_column])
    y = df[label_column]
    return X, y


def cross_validate_multiple_datasets(pattern: str, model, cv: int = 5):
    """Cross-validate a model across many datasets matched by pattern."""
    # Import here to avoid heavy dependency at module import time
    from sklearn.model_selection import cross_val_score  # type: ignore

    results = {}
    for fp in glob.glob(pattern):
        df = pd.read_csv(fp)
        X = df.drop(columns=["target"])
        y = df["target"]
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        name = fp.rsplit("/", 1)[-1].replace(".csv", "")
        results[name] = {
            "mean_accuracy": float(scores.mean()),
            "std_accuracy": float(scores.std()),
            "scores": scores.tolist(),
        }
    return results
