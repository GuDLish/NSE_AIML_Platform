"""
clustering.py
=============
Unsupervised segmentation of the NSE top-10 into three risk-return profiles
with K-Means, validated by silhouette score and an elbow curve.

Feature space (standardised):
    CAGR, annualised volatility, Sharpe ratio, maximum drawdown

Artefact: 17_req6_clusters.csv

Author : Prateek
License: MIT
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROFILE_ORDER = ["Defensive Compounders", "Balanced Core", "High-Beta Growth"]


def elbow_curve(X: np.ndarray, k_max: int = 6) -> pd.DataFrame:
    """Inertia and silhouette across candidate k, used to justify k=3."""
    rows = []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=25, random_state=42).fit(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, km.labels_),
        })
    return pd.DataFrame(rows)


def cluster_profiles(summary: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Assign each ticker to a risk-return profile, ordered by volatility."""
    cols = ["cagr", "annual_volatility", "sharpe", "max_drawdown"]
    X = StandardScaler().fit_transform(summary[cols].values)
    km = KMeans(n_clusters=k, n_init=25, random_state=42).fit(X)

    out = summary[cols].copy()
    out["cluster"] = km.labels_
    order = out.groupby("cluster")["annual_volatility"].mean().sort_values().index.tolist()
    naming = {c: PROFILE_ORDER[i] for i, c in enumerate(order)}
    out["profile"] = out["cluster"].map(naming)
    return out


def save_clusters(clusters: pd.DataFrame, path: str = "17_req6_clusters.csv") -> None:
    clusters.round(4).to_csv(path, index_label="ticker")


def allocation_suggestion(clusters: pd.DataFrame) -> pd.DataFrame:
    """Illustrative risk-budgeted weights: inverse-volatility inside each profile."""
    weights = []
    budget = {"Defensive Compounders": 0.40, "Balanced Core": 0.35, "High-Beta Growth": 0.25}
    for profile, group in clusters.groupby("profile"):
        inv = 1 / group["annual_volatility"]
        w = inv / inv.sum() * budget.get(profile, 1 / clusters["profile"].nunique())
        weights.append(w)
    return pd.concat(weights).rename("weight").to_frame().join(clusters[["profile"]])
