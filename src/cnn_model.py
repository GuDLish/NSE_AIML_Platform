"""
cnn_model.py
============
1D CNN (Conv1D + MaxPooling1D) for time-series feature extraction and trend
prediction, plus a GRU network for 30-day rolling volatility forecasting.

Why a CNN on prices?
--------------------
Convolution kernels act as learned technical patterns: a width-3 causal kernel
sliding over 60 days detects local motifs (breakouts, gaps, squeezes) far more
cheaply than a recurrent pass, and pooling gives translation invariance so the
same motif is recognised wherever it occurs in the window.

Author : Prateek
License: MIT
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Conv1D, Dense, Dropout, Flatten, GRU, Input, MaxPooling1D,
)
from tensorflow.keras.models import Sequential

from lstm_forecaster import SEQUENCE_LENGTH, evaluate, make_sequences


def build_cnn(timesteps: int, n_features: int) -> Sequential:
    """Conv1D -> MaxPool -> Conv1D -> MaxPool -> Dense regression head."""
    model = Sequential([
        Input(shape=(timesteps, n_features)),
        Conv1D(64, kernel_size=3, activation="relu", padding="causal"),
        MaxPooling1D(pool_size=2),
        Conv1D(32, kernel_size=3, activation="relu", padding="causal"),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="huber", metrics=["mae"])
    return model


def build_gru(timesteps: int, n_features: int) -> Sequential:
    """Stacked GRU used to forecast 30-day rolling volatility."""
    model = Sequential([
        Input(shape=(timesteps, n_features)),
        GRU(48, return_sequences=True),
        Dropout(0.2),
        GRU(24),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_cnn(features: pd.DataFrame, epochs: int = 40, verbose: int = 0) -> dict:
    """Train the CNN trend model and return hold-out metrics + predictions."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features.values.astype("float32"))
    X, y = make_sequences(scaled, SEQUENCE_LENGTH)
    split = int(len(X) * 0.85)

    model = build_cnn(SEQUENCE_LENGTH, scaled.shape[1])
    model.fit(X[:split], y[:split], epochs=epochs, batch_size=64, validation_split=0.1,
              verbose=verbose, callbacks=[EarlyStopping(patience=6, restore_best_weights=True)])

    def inverse(values, col=0):
        pad = np.zeros((len(values), scaled.shape[1]))
        pad[:, col] = np.asarray(values).ravel()
        return scaler.inverse_transform(pad)[:, col]

    y_pred = inverse(model.predict(X[split:], verbose=0).ravel())
    y_true = inverse(y[split:])
    return {"metrics": evaluate(y_true, y_pred), "y_true": y_true, "y_pred": y_pred, "model": model}


def train_gru_volatility(features: pd.DataFrame, epochs: int = 40, verbose: int = 0) -> dict:
    """Forecast the 30-day rolling standard deviation of returns with a GRU."""
    target_col = list(features.columns).index("vol30")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features.values.astype("float32"))
    X, y = make_sequences(scaled, SEQUENCE_LENGTH, target_col=target_col)
    split = int(len(X) * 0.85)

    model = build_gru(SEQUENCE_LENGTH, scaled.shape[1])
    model.fit(X[:split], y[:split], epochs=epochs, batch_size=64, validation_split=0.1,
              verbose=verbose, callbacks=[EarlyStopping(patience=6, restore_best_weights=True)])

    def inverse(values):
        pad = np.zeros((len(values), scaled.shape[1]))
        pad[:, target_col] = np.asarray(values).ravel()
        return scaler.inverse_transform(pad)[:, target_col]

    y_pred = inverse(model.predict(X[split:], verbose=0).ravel())
    y_true = inverse(y[split:])
    return {"metrics": evaluate(y_true, y_pred), "y_true": y_true, "y_pred": y_pred, "model": model}


def naive_baseline(y_true: np.ndarray) -> dict:
    """Persistence benchmark: tomorrow = today. Every model must beat this."""
    y_pred = np.concatenate([[y_true[0]], y_true[:-1]])
    return evaluate(y_true, y_pred)
