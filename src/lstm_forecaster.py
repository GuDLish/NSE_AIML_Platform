"""
lstm_forecaster.py
==================
Stacked LSTM price forecaster with a recursive 30-day sliding-window horizon.

Contract
--------
* input sequence length  : 60 trading days (configurable)
* model input shape      : 3D -> (samples, timesteps, features)
* forecast               : recursive — each prediction becomes the next input
* output                 : inverse-transformed back to real INR (Rs.) values
* artefact               : 30_days_forecast.csv with columns Day, Price (no index)

Author : Prateek
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

SEQUENCE_LENGTH = 60
FORECAST_HORIZON = 30


@dataclass
class ForecastResult:
    ticker: str
    current_price: float
    forecast: np.ndarray          # shape (FORECAST_HORIZON,) in INR
    metrics: dict
    y_true: np.ndarray
    y_pred: np.ndarray

    @property
    def day5(self) -> float:
        return float(self.forecast[4])

    @property
    def pct_change_day5(self) -> float:
        return float(self.forecast[4] / self.current_price - 1)


def make_sequences(scaled: np.ndarray, seq_len: int = SEQUENCE_LENGTH, target_col: int = 0):
    """Turn a 2D (time, features) matrix into 3D (samples, timesteps, features)."""
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i])       # 60 past days, all features
        y.append(scaled[i, target_col])       # next-day close
    return np.array(X), np.array(y)


def build_lstm(timesteps: int, n_features: int) -> Sequential:
    """Two stacked LSTM blocks with dropout regularisation, Huber loss."""
    model = Sequential([
        Input(shape=(timesteps, n_features)),      # 3D input: (samples, timesteps, features)
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="huber", metrics=["mae"])
    return model


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    direction = float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))))
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "direction_accuracy": direction,
    }


def forecast_ticker(
    features: pd.DataFrame,
    ticker: str,
    seq_len: int = SEQUENCE_LENGTH,
    horizon: int = FORECAST_HORIZON,
    epochs: int = 40,
    verbose: int = 0,
) -> ForecastResult:
    """Train on a chronological 85% split and roll a recursive forecast forward."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features.values.astype("float32"))
    X, y = make_sequences(scaled, seq_len)
    split = int(len(X) * 0.85)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    model = build_lstm(seq_len, scaled.shape[1])
    model.fit(
        X_train, y_train,
        epochs=epochs, batch_size=64, validation_split=0.1, verbose=verbose,
        callbacks=[EarlyStopping(patience=6, restore_best_weights=True)],
    )

    def inverse(values) -> np.ndarray:
        """Inverse-transform the close column back to INR."""
        pad = np.zeros((len(values), scaled.shape[1]))
        pad[:, 0] = np.asarray(values).ravel()
        return scaler.inverse_transform(pad)[:, 0]

    y_pred = inverse(model.predict(X_test, verbose=0).ravel())
    y_true = inverse(y_test)

    # --- recursive sliding-window forecast -------------------------------
    window = scaled[-seq_len:].copy()
    scaled_path = []
    for _ in range(horizon):
        next_scaled = float(model.predict(window.reshape(1, seq_len, -1), verbose=0)[0, 0])
        new_row = window[-1].copy()
        new_row[0] = next_scaled            # model output becomes next input
        window = np.vstack([window[1:], new_row])
        scaled_path.append(next_scaled)

    forecast = inverse(scaled_path)
    current = float(features["close"].iloc[-1])

    print(f"[{ticker}] current Rs.{current:,.2f} | day-5 Rs.{forecast[4]:,.2f} "
          f"| change {100 * (forecast[4] / current - 1):+.2f}%")

    return ForecastResult(ticker, current, forecast, evaluate(y_true, y_pred), y_true, y_pred)


def save_forecast_csv(result: ForecastResult, path: str = "30_days_forecast.csv") -> None:
    """Persist Day / Price columns with no row index, as required."""
    pd.DataFrame({
        "Day": np.arange(1, len(result.forecast) + 1),
        "Price": np.round(result.forecast, 2),
    }).to_csv(path, index=False)
