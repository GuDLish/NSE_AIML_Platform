"""
risk_classifier.py
==================
Random Forest classifier that flags an elevated probability of a >5% drawdown
over the next 21 trading sessions (roughly one calendar month).

Label:  y = 1 if  (close[t+21] / close[t] - 1) < -0.05  else 0

Artefact: risk_data_requirement.csv (feature matrix + labels used for training)

Author : Prateek
License: MIT
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

DOWNSIDE_THRESHOLD = -0.05
HORIZON_DAYS = 21


def label_downside(close: pd.Series, horizon: int = HORIZON_DAYS,
                   threshold: float = DOWNSIDE_THRESHOLD) -> pd.Series:
    forward = close.shift(-horizon) / close - 1
    return (forward < threshold).astype(int)


def build_dataset(market, build_features, tickers) -> pd.DataFrame:
    """Stack every ticker into one panel dataset with a binary downside label."""
    frames = []
    for ticker in tickers:
        features = build_features(market, ticker)
        labels = label_downside(market.close[ticker]).reindex(features.index)
        frame = features.drop(columns=["close"]).copy()
        frame["ticker"] = ticker
        frame["label"] = labels
        frames.append(frame.dropna())
    return pd.concat(frames)


def train_risk_model(dataset: pd.DataFrame, test_size: float = 0.25):
    X = dataset.drop(columns=["label", "ticker"])
    y = dataset["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=400, max_depth=10, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ).fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba > 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "base_rate": float(y.mean()),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    print(classification_report(y_test, pred, zero_division=0))
    importances = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
    )
    return model, metrics, importances


def live_probabilities(model, dataset: pd.DataFrame, tickers) -> pd.Series:
    """Latest observation per ticker scored by the trained forest."""
    out = {}
    for ticker in tickers:
        latest = dataset[dataset["ticker"] == ticker].drop(columns=["label", "ticker"]).iloc[[-1]]
        out[ticker] = float(model.predict_proba(latest)[0, 1])
    return pd.Series(out, name="downside_probability")


def save_risk_dataset(dataset: pd.DataFrame, path: str = "risk_data_requirement.csv") -> None:
    dataset.reset_index().rename(columns={"index": "Date"}).round(6).to_csv(path, index=False)
