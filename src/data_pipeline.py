"""
data_pipeline.py
================
Ingestion, cleaning, feature engineering and EDA artefacts for the
NSE Top-10 AI/ML Stock Market Platform.

Author : Prateek
License: MIT
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TICKERS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR",
    "ICICIBANK", "BHARTIARTL", "SBIN", "LT", "ITC",
]

FEATURE_COLUMNS = [
    "close", "ret", "ma20_rel", "vol30", "rsi", "mom10", "macd_hist", "bb_pctb",
]


@dataclass
class MarketData:
    """Container for the cleaned panel of OHLCV + engineered indicators."""

    close: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    returns: pd.DataFrame = field(init=False)
    ma20: pd.DataFrame = field(init=False)
    ma50: pd.DataFrame = field(init=False)
    vol30: pd.DataFrame = field(init=False)
    rsi14: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.returns = self.close.pct_change()
        self.ma20 = self.close.rolling(20).mean()
        self.ma50 = self.close.rolling(50).mean()
        self.vol30 = self.returns.rolling(30).std()
        self.rsi14 = self.close.apply(rsi)


# ----------------------------------------------------------------------------
# 1. Download / load
# ----------------------------------------------------------------------------
def download_from_yahoo(start: str = "2016-01-01", end: str | None = None) -> dict[str, pd.DataFrame]:
    """Pull raw OHLCV for the NSE top-10 using yfinance (``.NS`` suffix)."""
    import yfinance as yf

    symbols = [f"{t}.NS" for t in TICKERS]
    raw = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)
    out = {}
    for field_name, key in [("Close", "close"), ("High", "high"), ("Low", "low"), ("Volume", "volume")]:
        df = raw[field_name].copy()
        df.columns = [c.replace(".NS", "") for c in df.columns]
        out[key] = df[TICKERS]
    return out


def load_local(folder: str = "data") -> dict[str, pd.DataFrame]:
    """Load the committed CSV snapshots so the repo runs fully offline."""
    files = {
        "close": "01_close_prices.csv",
        "high": "02_high_prices.csv",
        "low": "03_low_prices.csv",
        "volume": "04_volume.csv",
    }
    return {
        k: pd.read_csv(os.path.join(folder, v), parse_dates=["Date"], index_col="Date")[TICKERS]
        for k, v in files.items()
    }


# ----------------------------------------------------------------------------
# 2. Cleaning & preprocessing
# ----------------------------------------------------------------------------
def clean(frames: dict[str, pd.DataFrame]) -> MarketData:
    """Reindex to a continuous business-day calendar and forward/back fill gaps.

    Market holidays and suspended sessions create irregular gaps; a continuous
    calendar keeps every rolling window (20/30/50/60) genuinely time-aligned
    across all ten tickers.
    """
    idx = pd.bdate_range(
        min(f.index.min() for f in frames.values()),
        max(f.index.max() for f in frames.values()),
    )
    cleaned = {k: v.reindex(idx).ffill().bfill() for k, v in frames.items()}
    return MarketData(**cleaned)


def quality_report(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Missing-value and coverage audit, printed before/after cleaning."""
    rows = []
    for name, df in frames.items():
        rows.append({
            "frame": name,
            "rows": len(df),
            "missing": int(df.isna().sum().sum()),
            "missing_pct": round(100 * df.isna().sum().sum() / df.size, 4),
            "start": str(df.index.min().date()),
            "end": str(df.index.max().date()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 3. Technical indicators / feature engineering
# ----------------------------------------------------------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI:  RSI = 100 − 100 / (1 + RS),  RS = avg gain / avg loss."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def build_features(md: MarketData, ticker: str) -> pd.DataFrame:
    """Per-ticker feature matrix: momentum, trend, mean-reversion, volatility."""
    close, high, low, volume = md.close[ticker], md.high[ticker], md.low[ticker], md.volume[ticker]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()

    frame = pd.DataFrame({
        "close": close,
        "ret": close.pct_change(),
        "ma20_rel": close / md.ma20[ticker] - 1,
        "ma50_rel": close / md.ma50[ticker] - 1,
        "vol30": md.vol30[ticker],
        "rsi": md.rsi14[ticker] / 100,
        "mom10": close.pct_change(10),
        "mom21": close.pct_change(21),
        "macd_hist": (macd - macd_signal) / close,
        "bb_pctb": (close - (bb_mid - 2 * bb_std)) / (4 * bb_std),
        "atr": ((high - low) / close).rolling(14).mean(),
        "vol_z": (volume - volume.rolling(60).mean()) / volume.rolling(60).std(),
    })
    return frame.dropna()


# ----------------------------------------------------------------------------
# 4. EDA summary
# ----------------------------------------------------------------------------
def summary_stats(md: MarketData, risk_free: float = 0.065) -> pd.DataFrame:
    """CAGR, annualised volatility, Sharpe, max drawdown, VaR(95) per ticker."""
    rows = []
    years = (md.close.index[-1] - md.close.index[0]).days / 365.25
    for t in TICKERS:
        r = md.returns[t].dropna()
        total = md.close[t].iloc[-1] / md.close[t].iloc[0] - 1
        cagr = (1 + total) ** (1 / years) - 1
        ann_vol = r.std() * np.sqrt(252)
        rows.append({
            "ticker": t,
            "cagr": cagr,
            "annual_volatility": ann_vol,
            "sharpe": (r.mean() * 252 - risk_free) / ann_vol,
            "max_drawdown": ((md.close[t] / md.close[t].cummax()) - 1).min(),
            "var_95": np.percentile(r, 5),
            "skew": r.skew(),
            "kurtosis": r.kurtosis(),
        })
    return pd.DataFrame(rows).set_index("ticker")


def export_clean_csvs(md: MarketData, folder: str = "data") -> None:
    """Write the nine canonical cleaned datasets used by the dashboard."""
    os.makedirs(folder, exist_ok=True)
    mapping = {
        "01_close_prices.csv": md.close, "02_high_prices.csv": md.high,
        "03_low_prices.csv": md.low, "04_volume.csv": md.volume,
        "05_ma_20.csv": md.ma20, "06_ma_50.csv": md.ma50,
        "07_volatility_30.csv": md.vol30, "08_rsi.csv": md.rsi14,
        "09_daily_returns.csv": md.returns,
    }
    for name, df in mapping.items():
        df.round(6).to_csv(os.path.join(folder, name), index_label="Date")


if __name__ == "__main__":
    frames = load_local()
    print(quality_report(frames).to_string(index=False))
    market = clean(frames)
    export_clean_csvs(market)
    print(summary_stats(market).round(4).to_string())
