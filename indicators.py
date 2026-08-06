import pandas as pd

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def compute_bollinger(series, window=20, num_std=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    pct_b = (series - lower) / (upper - lower)
    return upper, lower, pct_b

def engineer_features(df, include_target=True):
    df = df.copy()
    df['Returns'] = df['Close'].pct_change()
    df['Returns_5d'] = df['Close'].pct_change(5)
    df['Returns_10d'] = df['Close'].pct_change(10)
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['MACD'], df['MACD_Signal'] = compute_macd(df['Close'])
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    df['BB_Upper'], df['BB_Lower'], df['BB_PctB'] = compute_bollinger(df['Close'])
    df['ATR'] = compute_atr(df, 14)
    df['ATR_Pct'] = df['ATR'] / df['Close'] * 100
    df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_Diff'] = (df['SMA_20'] - df['SMA_50']) / df['SMA_50']
    df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['Volatility_20'] = df['Returns'].rolling(20).std()
    df['Volatility_5'] = df['Returns'].rolling(5).std()
    df['Return_Lag1'] = df['Returns'].shift(1)
    df['Return_Lag2'] = df['Returns'].shift(2)
    df['Return_Lag3'] = df['Returns'].shift(3)

    if 'VIX_Close' not in df.columns:
        df['VIX_Close'] = 20.0
    if 'VIX_MA_5' not in df.columns:
        df['VIX_MA_5'] = df['VIX_Close'].rolling(window=5, min_periods=1).mean()
    else:
        df['VIX_MA_5'] = df['VIX_MA_5'].fillna(df['VIX_Close'])

    if include_target:
        df['Target_Return'] = df['Returns'].shift(-1)
        df['Target_Direction'] = (df['Target_Return'] > 0).astype(int)
        return df.dropna()
    else:
        feature_cols = [
            'Returns', 'Returns_5d', 'Returns_10d', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_PctB', 'ATR_Pct', 'Volume_Ratio', 'SMA_Diff', 'Momentum_5', 'Momentum_10',
            'Volatility_20', 'Volatility_5', 'Return_Lag1', 'Return_Lag2', 'Return_Lag3',
            'VIX_Close', 'VIX_MA_5'
        ]
        return df.dropna(subset=feature_cols)
