import pandas as pd

def add_features(stock_data: pd.DataFrame) -> pd.DataFrame:
    df = stock_data.copy()
    df["MA3"] = df["CLOSE"].rolling(window=3).mean()
    df["MA5"] = df["CLOSE"].rolling(window=5).mean()
    df["EMA10"] = df["CLOSE"].ewm(span=10, adjust=False).mean()
    df["STD_5"] = df["CLOSE"].rolling(window=5).std()
    df["RETURN"] = df["CLOSE"].pct_change()
    df = df.dropna()
    return df