import yfinance as yf
import pandas as pd

def load_stock_data(ticker, start_date, end_date):
    """Downloads stock data from Yahoo Finance"""
    df = yf.download(
        ticker, 
        start=start_date, 
        end=end_date,
        auto_adjust=True
    )

    # Flatten multi-index columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    df['Date'] = pd.to_datetime(df['Date'])
    
    return df