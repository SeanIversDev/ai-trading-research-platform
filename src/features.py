import pandas as pd

def create_features(df):
    df = df.copy()

    df['return_1d'] = df['Close'].pct_change()
    df['sma_10'] = df['Close'].rolling(10).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['volatility_10'] = df['return_1d'].rolling(10).std()


    # Target: 1 if tomorrow is higher, 0 otherwise
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    df = df.dropna()

    features = [
        'return_1d',
        'sma_10',
        'sma_50',
        'volatility_10',
        'Volume'
    ]

    return df, features