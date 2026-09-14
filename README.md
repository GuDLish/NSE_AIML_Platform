# 📈 NSE Top-10 AI/ML Stock Market Platform

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-2.1%2B-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?logo=googlecolab&logoColor=white)

End-to-end deep-learning and machine-learning research platform on **10 years of daily data (2016-02-16 → 2026-02-13, 2,609 business days)** for the ten largest NSE names:

`RELIANCE · TCS · HDFCBANK · INFY · HINDUNILVR · ICICIBANK · BHARTIARTL · SBIN · LT · ITC`

It covers the full data-science lifecycle: cleaning → EDA → feature engineering → unsupervised clustering → supervised risk classification → LSTM / 1D-CNN / GRU deep learning → evaluation → business insight.

---

## 🏗️ Architecture

```text
              ┌──────────────────────────┐
raw OHLCV ───▶│  data_pipeline.py        │  business-day reindex, ffill,
   CSV /      │  clean · features · EDA  │  RSI/MACD/BB/momentum, z-scores
  yfinance    └────────────┬─────────────┘
                           │  tidy panel (8 features × 10 tickers)
        ┌──────────────────┼───────────────────┬────────────────────┐
        ▼                  ▼                   ▼                    ▼
┌───────────────┐  ┌───────────────┐  ┌────────────────┐  ┌──────────────────┐
│ lstm_         │  │ cnn_model.py  │  │ clustering.py  │  │ risk_classifier  │
│ forecaster.py │  │ Conv1D+Pool   │  │ K-Means (k=3)  │  │ RandomForest     │
│ 60→1 stacked  │  │ trend model   │  │ risk-return    │  │ P(drop > 5%)     │
│ LSTM, 30-day  │  │ + GRU σ model │  │ profiles       │  │ over 21 sessions │
│ recursive     │  └───────┬───────┘  └────────┬───────┘  └─────────┬────────┘
└───────┬───────┘          │                   │                    │
        └──────────────────┴─────────┬─────────┴────────────────────┘
                                     ▼
                         ┌───────────────────────┐
                         │ main.py → outputs/    │
                         │ CSVs + app_bundle.json│
                         └───────────┬───────────┘
                                     ▼
                        interactive React terminal UI
```

---

## 🧪 Pipeline parts

| Part | Stage | Output |
|------|-------|--------|
| 1 | Data loading & cleaning (missing dates, ffill, dedupe) | `01–04` cleaned series |
| 2 | Technical indicators (MA20/50, σ30, RSI14) | `05–08` |
| 3 | Daily returns & distributional stats | `09_daily_returns.csv` |
| 4 | EDA: trends, volume, correlation heatmap | dashboard charts |
| 5 | Feature engineering & MinMax scaling | 8-feature matrix |
| 6 | K-Means clustering into 3 risk-return profiles | `17_req6_clusters.csv` |
| 7 | LSTM 60-day sequence forecaster + recursive 30-day path | `30_days_forecast.csv` |
| 8 | 1D CNN trend model & GRU volatility model | `results.csv` |
| 9 | Random Forest downside-risk classifier | `risk_data_requirement.csv` |
| 10 | Evaluation, benchmarks & executive conclusions | `model_benchmarks.csv` |

---

## 🧮 Mathematics

**Daily log/simple return**

```
r_t = P_t / P_{t-1} − 1
```

**Annualised volatility (30-day rolling)**

```
σ_ann = std(r_{t−29..t}) × √252
```

**Sharpe ratio (rf = 0)**

```
S = (mean(r) × 252) / (std(r) × √252)
```

**RSI (Wilder, 14)**

```
RS  = EMA(gain, 14) / EMA(loss, 14)
RSI = 100 − 100 / (1 + RS)
```

**Max drawdown**

```
MDD = min_t ( P_t / max_{s ≤ t} P_s − 1 )
```

**MinMax scaling (inverse-transformed back to ₹ after prediction)**

```
x' = (x − min) / (max − min)
```

**LSTM input tensor** — `(samples, timesteps=60, features=8)`; recursive forecasting feeds each prediction back as the newest timestep for 30 steps.

**Evaluation**

```
RMSE = √( Σ(y − ŷ)² / n )      MAE = Σ|y − ŷ| / n
R²   = 1 − SS_res / SS_tot      MAPE = 100/n · Σ|(y − ŷ)/y|
DirAcc = mean( sign(Δy) == sign(Δŷ) )
```

---

## 📊 Model benchmarks

Chronological 70/15/15 split; metrics on the unseen 15% hold-out. Full per-ticker numbers in [`data/model_benchmarks.csv`](data/model_benchmarks.csv).

| Model | Task | Key metrics |
|-------|------|-------------|
| Stacked LSTM (64→32) | next-close from 60×8 window | RMSE, MAE, R², direction accuracy |
| 1D CNN (Conv1D 64 + MaxPool) | trend / next-close | compared head-to-head with LSTM |
| GRU (32) | 30-day rolling σ | RMSE on annualised volatility |
| Naive random walk | baseline `ŷ_t = y_{t−1}` | reference floor every model must beat |
| K-Means (k=3) | risk-return profiles | silhouette-selected k |
| Random Forest (400 trees) | P(drop > 5% in 21d) | accuracy, precision, recall, F1, ROC-AUC |

---

## 🚀 Run it

### Google Colab
```python
!pip install -r requirements.txt
```
Then open `NSE_Stock_AIML_Master_Pipeline.ipynb` and **Runtime → Run all**. It executes parts 1–10 top to bottom and writes every CSV into `outputs/`.

### Local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/main.py --ticker RELIANCE --epochs 25   # one stock
python src/main.py --all --epochs 25               # full universe
```

---

## 📁 Repository layout

```text
NSE-Stock-AIML-Platform/
├── NSE_Stock_AIML_Master_Pipeline.ipynb   # documented, executable, parts 1–10
├── src/
│   ├── data_pipeline.py      # load, clean, indicators, EDA
│   ├── lstm_forecaster.py    # 60-day sequences, recursive 30-day forecast
│   ├── cnn_model.py          # Conv1D trend model + GRU volatility model
│   ├── clustering.py         # K-Means risk-return profiles
│   ├── risk_classifier.py    # Random Forest downside classifier
│   └── main.py               # orchestration CLI
├── data/                     # cleaned series 01–09 + model outputs
├── outputs/app_bundle.json   # everything the dashboard renders
├── requirements.txt · .gitignore · LICENSE
```

---

## 📦 Deliverables

- `30_days_forecast.csv` — Ticker · Day · Price for every recursive LSTM path
- `results.csv` — actual vs predicted close on the hold-out window
- `risk_data_requirement.csv` — engineered features + >5% downside labels
- `17_req6_clusters.csv` — K-Means risk-return profile per ticker

---

## ⚠️ Disclaimer

Research and educational project. Recursive forecasts compound model error and direction accuracy hovers near 50% — nothing here is investment advice.

## 📄 License

MIT © Prateek — see [LICENSE](LICENSE).
