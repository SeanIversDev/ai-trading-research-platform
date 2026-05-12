import streamlit as st
import plotly.express as px

from src.data_loader import load_stock_data
from src.features import create_features
from src.models import train_models
from src.backtester import run_backtest


st.set_page_config(
    page_title="AI Trading Research Platform",
    layout="wide"
)

st.title("AI Trading Research Platform")

ticker = st.sidebar.text_input("Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

run_button = st.sidebar.button("Run Analysis")


if run_button:
    try:
        df = load_stock_data(ticker, start_date, end_date)
        featured_df, features = create_features(df)
        results, test_df = train_models(featured_df, features)

        st.session_state["df"] = df
        st.session_state["results"] = results
        st.session_state["test_df"] = test_df
        st.session_state["ticker"] = ticker

    except Exception as e:
        st.error(f"Error: {e}")


if "results" in st.session_state:
    df = st.session_state["df"]
    results = st.session_state["results"]
    test_df = st.session_state["test_df"]
    ticker = st.session_state["ticker"]

    st.subheader(f"{ticker} Price Data")
    st.dataframe(df.tail())

    fig_price = px.line(
        df,
        x="Date",
        y="Close",
        title=f"{ticker} Closing Price"
    )

    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Logistic Regression Accuracy",
            f"{results['Logistic Regression']['accuracy']:.2%}"
        )

    with col2:
        st.metric(
            "Random Forest Accuracy",
            f"{results['Random Forest']['accuracy']:.2%}"
        )

    selected_model = st.selectbox(
        "Choose Model for Backtest",
        list(results.keys())
    )

    predictions = results[selected_model]["predictions"]

    try:
        backtest_df, metrics = run_backtest(test_df, predictions)

        st.subheader(f"Backtest Results: {selected_model}")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Strategy Return", f"{metrics['Strategy Return']:.2%}")
        col2.metric("Buy & Hold Return", f"{metrics['Buy & Hold Return']:.2%}")
        col3.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
        col4.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")

        fig_backtest = px.line(
            backtest_df,
            x="Date",
            y=["market_cumulative", "strategy_cumulative"],
            title="Strategy vs Buy & Hold"
        )

        st.plotly_chart(fig_backtest, use_container_width=True)

        st.subheader("Backtest Data")
        st.dataframe(backtest_df.tail())

    except Exception as e:
        st.error(f"Backtest error: {e}")