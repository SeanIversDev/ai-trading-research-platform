import numpy as np

def run_backtest(test_df, predictions):

    df = test_df.copy()

    df["prediction"] = predictions

    # Market returns
    df["market_return"] = df["Close"].pct_change()

    # Shift predictions to avoid look-ahead bias
    df["strategy_return"] = (
        df["prediction"].shift(1) * df["market_return"]
    )

    # Remove NaNs safely
    df = df.dropna(subset=["market_return", "strategy_return"])

    # Safety check
    if df.empty:
        raise ValueError("Backtest dataframe is empty after dropping NaNs.")

    # Cumulative performance
    df["market_cumulative"] = (
        1 + df["market_return"]
    ).cumprod()

    df["strategy_cumulative"] = (
        1 + df["strategy_return"]
    ).cumprod()

    # Metrics
    total_return = df["strategy_cumulative"].iloc[-1] - 1

    buy_hold_return = (
        df["market_cumulative"].iloc[-1] - 1
    )

    sharpe_ratio = (
        df["strategy_return"].mean()
        / df["strategy_return"].std()
    ) * np.sqrt(252)

    max_drawdown = (
        df["strategy_cumulative"]
        / df["strategy_cumulative"].cummax()
        - 1
    ).min()

    metrics = {
        "Strategy Return": total_return,
        "Buy & Hold Return": buy_hold_return,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown
    }

    return df, metrics