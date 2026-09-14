"""
main.py
=======
End-to-end orchestrator for the NSE Top-10 AI/ML Stock Market Platform.

Run:
    python src/main.py --ticker RELIANCE --epochs 40
    python src/main.py --all

Pipeline stages
    1. Load + clean          -> data_pipeline.clean
    2. Feature engineering   -> data_pipeline.build_features
    3. EDA summary           -> data_pipeline.summary_stats
    4. LSTM forecast         -> lstm_forecaster.forecast_ticker
    5. CNN + GRU             -> cnn_model.train_cnn / train_gru_volatility
    6. K-Means clustering    -> clustering.cluster_profiles
    7. Risk classifier       -> risk_classifier.train_risk_model
    8. Artefacts             -> CSV deliverables

Author : Prateek
License: MIT
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from clustering import cluster_profiles, elbow_curve, save_clusters
from cnn_model import naive_baseline, train_cnn, train_gru_volatility
from data_pipeline import (
    TICKERS, FEATURE_COLUMNS, build_features, clean, export_clean_csvs,
    load_local, quality_report, summary_stats,
)
from lstm_forecaster import forecast_ticker, save_forecast_csv
from risk_classifier import build_dataset, live_probabilities, save_risk_dataset, train_risk_model

OUTPUT_DIR = "outputs"


def run(tickers: list[str], epochs: int = 40) -> dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 78, "\nSTAGE 1/8  Load & audit raw data", sep="")
    frames = load_local()
    print(quality_report(frames).to_string(index=False))

    print("\nSTAGE 2/8  Clean (business-day reindex + forward fill)")
    market = clean(frames)
    export_clean_csvs(market, folder="data")

    print("\nSTAGE 3/8  EDA summary statistics")
    summary = summary_stats(market)
    print(summary.round(4).to_string())

    print("\nSTAGE 4/8  K-Means risk-return clustering")
    print(elbow_curve(
        __import__("sklearn.preprocessing", fromlist=["StandardScaler"])
        .StandardScaler().fit_transform(summary[["cagr", "annual_volatility", "sharpe", "max_drawdown"]])
    ).round(4).to_string(index=False))
    clusters = cluster_profiles(summary)
    save_clusters(clusters, os.path.join(OUTPUT_DIR, "17_req6_clusters.csv"))
    print(clusters[["profile"]].to_string())

    print("\nSTAGE 5/8  Random Forest downside-risk classifier")
    risk_dataset = build_dataset(market, build_features, TICKERS)
    rf, rf_metrics, importances = train_risk_model(risk_dataset)
    save_risk_dataset(risk_dataset, os.path.join(OUTPUT_DIR, "risk_data_requirement.csv"))
    print(live_probabilities(rf, risk_dataset, TICKERS).round(3).to_string())

    all_forecasts, all_results, benchmarks = [], [], []
    for ticker in tickers:
        print(f"\nSTAGE 6/8  Deep learning for {ticker}")
        features = build_features(market, ticker)[FEATURE_COLUMNS]

        lstm = forecast_ticker(features, ticker, epochs=epochs)
        cnn = train_cnn(features, epochs=epochs)
        gru = train_gru_volatility(features, epochs=epochs)
        naive = naive_baseline(lstm.y_true)

        all_forecasts.append(pd.DataFrame({
            "Ticker": ticker,
            "Day": np.arange(1, len(lstm.forecast) + 1),
            "Price": np.round(lstm.forecast, 2),
        }))
        all_results.append(pd.DataFrame({
            "Ticker": ticker, "actual": lstm.y_true, "predicted": lstm.y_pred,
        }))
        for name, m in [("LSTM", lstm.metrics), ("CNN", cnn["metrics"]),
                        ("GRU_volatility", gru["metrics"]), ("Naive", naive)]:
            benchmarks.append({"ticker": ticker, "model": name, **m})

    print("\nSTAGE 7/8  Persist deliverables")
    pd.concat(all_forecasts).to_csv(os.path.join(OUTPUT_DIR, "30_days_forecast.csv"), index=False)
    pd.concat(all_results).round(4).to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)
    bench = pd.DataFrame(benchmarks)
    bench.round(4).to_csv(os.path.join(OUTPUT_DIR, "model_benchmarks.csv"), index=False)
    if len(tickers) == 1:
        save_forecast_csv(lstm, os.path.join(OUTPUT_DIR, "30_days_forecast_single.csv"))

    print("\nSTAGE 8/8  Executive summary")
    print(bench.groupby("model")[["rmse", "mae", "r2", "direction_accuracy"]].mean().round(4).to_string())
    with open(os.path.join(OUTPUT_DIR, "risk_metrics.json"), "w") as fh:
        json.dump({"random_forest": rf_metrics,
                   "feature_importance": importances.round(4).to_dict()}, fh, indent=2)
    print(f"\nAll artefacts written to ./{OUTPUT_DIR}/")
    return {"benchmarks": bench, "clusters": clusters, "risk_metrics": rf_metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSE Top-10 AI/ML stock platform")
    parser.add_argument("--ticker", default="RELIANCE", choices=TICKERS)
    parser.add_argument("--all", action="store_true", help="run every ticker")
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    run(TICKERS if args.all else [args.ticker], epochs=args.epochs)
