# AI Trading Research Platform

A Streamlit web app for researching ML-based trading strategies on any stock. Enter a ticker and date range, train two classifiers, and compare strategy performance against buy & hold via backtesting.

## Features

- Fetches historical price data from Yahoo Finance for any ticker
- Engineers technical features (returns, moving averages, volatility)
- Trains and compares Logistic Regression and Random Forest classifiers
- Backtests predictions against buy & hold with Sharpe Ratio and max drawdown metrics
- Interactive charts via Plotly

## How It Works

### 1. Data Loading
Yahoo Finance data is pulled via `yfinance` for the selected ticker and date range.

### 2. Feature Engineering
| Feature | Description |
|---------|-------------|
| `return_1d` | Daily percentage return |
| `sma_10` | 10-day simple moving average |
| `sma_50` | 50-day simple moving average |
| `volatility_10` | 10-day rolling standard deviation of returns |
| `Volume` | Trading volume |

The target variable is binary: `1` if tomorrow's close is higher than today's, `0` otherwise.

### 3. Models
Both models use an 80/20 chronological train/test split (no shuffling, to preserve time-series order).

- **Logistic Regression** — trained on `StandardScaler`-normalized features
- **Random Forest** — 100 estimators, trained on raw features

### 4. Backtesting
The selected model's predictions are used to simulate a long-only strategy (hold when prediction = 1, stay out when prediction = 0). Results are compared against a simple buy & hold baseline.

| Metric | Description |
|--------|-------------|
| Strategy Return | Cumulative return of the model-driven strategy |
| Buy & Hold Return | Cumulative return of holding the stock throughout |
| Sharpe Ratio | Annualized risk-adjusted return (√252 scaling) |
| Max Drawdown | Largest peak-to-trough decline during the period |

## Project Structure

```
ai-trading-research-platform/
├── app.py                  # Streamlit UI
├── requirements.txt
└── src/
    ├── data_loader.py      # Yahoo Finance data fetching
    ├── features.py         # Technical indicator engineering
    ├── models.py           # Logistic Regression & Random Forest training
    └── backtester.py       # Strategy simulation and metrics
```

## Setup & Running

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the sidebar, enter a ticker (e.g. `AAPL`), select a date range, and click **Run Analysis**.

## Dependencies

- [Streamlit](https://streamlit.io) — web UI
- [yfinance](https://github.com/ranaroussi/yfinance) — market data
- [scikit-learn](https://scikit-learn.org) — ML models
- [Plotly](https://plotly.com/python/) — interactive charts
- pandas, numpy
