import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import sqlite3
import json
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# OPSIYONEL ML KÜTÜPHANELERİ — graceful fallback
# ---------------------------------------------------------
_HAS_LIGHTGBM = False
_HAS_XGBOOST = False
_HAS_SKLEARN = False

try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:
    pass

try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except ImportError:
    pass

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
    _HAS_SKLEARN = True
except ImportError:
    pass

# ---------------------------------------------------------
# SAYFA YAPILANDIRMASI & MODERN CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Borsa Panel",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FILE = "son_analiz.csv"
DB_FILE = "avanza_quant.db"

st.markdown("""
    <style>
    /* Ana konteyner */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
        max-width: 1400px;
    }

    /* Sidebar karanlık tema */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] label {
        color: #e0e0e0 !important;
    }

    /* Başlık stilleri */
    h1 {
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        color: #e8e8e8;
        font-weight: 600;
    }

    /* Tab stilleri */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 10px 10px 0 0;
        border: 1px solid #333;
        background: #1a1a2e;
        color: #aaa;
        font-weight: 500;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3a5f, #2d5a87);
        color: #fff !important;
        border-color: #00d4ff;
        box-shadow: 0 -2px 8px rgba(0, 212, 255, 0.15);
    }

    /* Metric kartları — kompakt */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 0.4rem 0.6rem;
        border-radius: 0.5rem;
        border-left: 2px solid #00d4ff;
        box-shadow: 0 1px 4px rgba(0,0,0,0.2);
    }
    [data-testid="stMetric"] label {
        color: #8ab4f8 !important;
        font-size: 0.65rem !important;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1rem !important;
        font-weight: 700;
    }
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 0.7rem !important;
    }

    /* Hisse kartı */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        gap: 0.3rem;
    }

    /* Buton stilleri */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.2s;
        border: none;
        background: linear-gradient(135deg, #00d4ff, #7b2ff7);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(0, 212, 255, 0.3);
    }

    /* Selectbox stilleri */
    [data-baseweb="select"] > div {
        background: #1a1a2e;
        border-color: #333;
        border-radius: 8px;
    }

    /* Info/Warning/Success/Error kutuları */
    .stAlert {
        border-radius: 10px;
        border: 1px solid;
    }
    .stAlert [data-testid="stAlertContentInfo"] {
        background: rgba(0, 212, 255, 0.08);
        border-color: rgba(0, 212, 255, 0.3);
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Divider */
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #333, transparent);
        margin: 1.5rem 0;
    }

    /* Scroll bar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0f0f1e;
    }
    ::-webkit-scrollbar-thumb {
        background: #333;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }

    /* Caption */
    .stCaption {
        color: #666 !important;
        font-size: 0.82rem !important;
    }

    /* Sinyal rozeti */
    .signal-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .signal-buy { background: #1b4332; color: #52d681; border: 1px solid #52d681; }
    .signal-sell { background: #4a1a1a; color: #ff6b6b; border: 1px solid #ff6b6b; }
    .signal-hold { background: #3d3a0a; color: #ffd93d; border: 1px solid #ffd93d; }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)


# ===================================================================
# 1. SQLite VERİTABANI KATMANI
# ===================================================================
class DatabaseManager:
    """SQLite ile tahmin, backtest ve metrik geçmişi yönetimi."""

    def __init__(self, db_path=DB_FILE):
        self.db_path = db_path
        self._init_tables()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_tables(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tarih TEXT, kod TEXT, sirket_adi TEXT,
                    son_fiyat REAL, sinyal TEXT,
                    tahmin_1g REAL, tahmin_2g REAL, tahmin_3g REAL,
                    rsi_15m REAL, vwap REAL, strateji TEXT,
                    analiz_zamani TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date TEXT, kod TEXT,
                    initial_capital REAL, final_capital REAL,
                    total_return REAL, sharpe REAL, sortino REAL,
                    max_drawdown REAL, cagr REAL, win_rate REAL,
                    total_trades INTEGER, profit_factor REAL,
                    volatility REAL,
                    equity_curve TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS backtest_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER, kod TEXT,
                    entry_date TEXT, entry_price REAL,
                    exit_date TEXT, exit_price REAL,
                    shares INTEGER, pnl REAL, return_pct REAL,
                    side TEXT, reason TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tarih TEXT, kod TEXT,
                    predicted_direction INTEGER,
                    probability REAL,
                    model_name TEXT,
                    accuracy REAL,
                    feature_importance TEXT
                )
            """)

    def save_prediction(self, row_dict):
        with self._conn() as c:
            c.execute("""
                INSERT INTO predictions
                (tarih, kod, sirket_adi, son_fiyat, sinyal,
                 tahmin_1g, tahmin_2g, tahmin_3g,
                 rsi_15m, vwap, strateji, analiz_zamani)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                datetime.now().strftime("%Y-%m-%d"),
                row_dict.get('Kod', ''),
                row_dict.get('Şirket Adı', ''),
                row_dict.get('Son Fiyat (SEK)', 0),
                row_dict.get('Sinyal', ''),
                row_dict.get('1. Gün Tahmin (%)', 0),
                row_dict.get('2. Gün Tahmin (%)', 0),
                row_dict.get('3. Gün Tahmin (%)', 0),
                row_dict.get('RSI (15m)', 0),
                row_dict.get('VWAP', 0),
                row_dict.get('Strateji', ''),
                row_dict.get('Analiz Zamanı', '')
            ))

    def save_backtest_run(self, result):
        with self._conn() as c:
            cur = c.execute("""
                INSERT INTO backtest_runs
                (run_date, kod, initial_capital, final_capital,
                 total_return, sharpe, sortino, max_drawdown, cagr,
                 win_rate, total_trades, profit_factor, volatility,
                 equity_curve)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                result['kod'],
                result['initial_capital'],
                result['final_capital'],
                result['total_return'],
                result['sharpe'],
                result['sortino'],
                result['max_drawdown'],
                result['cagr'],
                result['win_rate'],
                result['total_trades'],
                result['profit_factor'],
                result['volatility'],
                json.dumps(result['equity_curve'])
            ))
            return cur.lastrowid

    def save_backtest_trade(self, run_id, trade):
        with self._conn() as c:
            c.execute("""
                INSERT INTO backtest_trades
                (run_id, kod, entry_date, entry_price,
                 exit_date, exit_price, shares, pnl, return_pct, side, reason)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                run_id, trade['kod'], trade['entry_date'], trade['entry_price'],
                trade['exit_date'], trade['exit_price'], trade['shares'],
                trade['pnl'], trade['return_pct'], trade['side'], trade['reason']
            ))

    def save_ml_prediction(self, kod, direction, probability, model_name, accuracy, feature_importance):
        with self._conn() as c:
            c.execute("""
                INSERT INTO ml_predictions
                (tarih, kod, predicted_direction, probability,
                 model_name, accuracy, feature_importance)
                VALUES (?,?,?,?,?,?,?)
            """, (
                datetime.now().strftime("%Y-%m-%d %H:%M"), kod,
                direction, probability, model_name, accuracy,
                json.dumps(feature_importance)
            ))

    def get_prediction_history(self, kod=None, limit=100):
        with self._conn() as c:
            if kod:
                df = pd.read_sql_query(
                    "SELECT * FROM predictions WHERE kod=? ORDER BY id DESC LIMIT ?",
                    c, params=(kod, limit))
            else:
                df = pd.read_sql_query(
                    "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", c, params=(limit,))
            return df

    def get_backtest_history(self, kod=None, limit=50):
        with self._conn() as c:
            if kod:
                df = pd.read_sql_query(
                    "SELECT * FROM backtest_runs WHERE kod=? ORDER BY id DESC LIMIT ?",
                    c, params=(kod, limit))
            else:
                df = pd.read_sql_query(
                    "SELECT * FROM backtest_runs ORDER BY id DESC LIMIT ?",
                    c, params=(limit,))
            return df

    def get_ml_history(self, kod=None, limit=50):
        with self._conn() as c:
            if kod:
                df = pd.read_sql_query(
                    "SELECT * FROM ml_predictions WHERE kod=? ORDER BY id DESC LIMIT ?",
                    c, params=(kod, limit))
            else:
                df = pd.read_sql_query(
                    "SELECT * FROM ml_predictions ORDER BY id DESC LIMIT ?",
                    c, params=(limit,))
            return df

    def get_trades_for_run(self, run_id):
        with self._conn() as c:
            return pd.read_sql_query(
                "SELECT * FROM backtest_trades WHERE run_id=? ORDER BY id",
                c, params=(run_id,))


db = DatabaseManager()


# ===================================================================
# 2. TEKNİK ANALİZ — FEATURE ENGINEERING
# ===================================================================
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
    """
    Geçmişe bakarak (lookahead yok) ML feature'ları üretir.
    include_target=True  -> training için label sütunları ekler (son satır dropna ile düşer).
    include_target=False -> inference için son satırı korur (label yok).
    """
    df = df.copy()

    # Temel getiriler
    df['Returns'] = df['Close'].pct_change()
    df['Returns_5d'] = df['Close'].pct_change(5)
    df['Returns_10d'] = df['Close'].pct_change(10)

    # RSI
    df['RSI'] = compute_rsi(df['Close'], 14)

    # MACD
    df['MACD'], df['MACD_Signal'] = compute_macd(df['Close'])
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands
    df['BB_Upper'], df['BB_Lower'], df['BB_PctB'] = compute_bollinger(df['Close'])

    # ATR
    df['ATR'] = compute_atr(df, 14)
    df['ATR_Pct'] = df['ATR'] / df['Close'] * 100

    # Volume
    df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']

    # Moving averages
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_Diff'] = (df['SMA_20'] - df['SMA_50']) / df['SMA_50']

    # Momentum
    df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1

    # Volatilite
    df['Volatility_20'] = df['Returns'].rolling(20).std()
    df['Volatility_5'] = df['Returns'].rolling(5).std()

    # Lag features (geçmiş getiri)
    df['Return_Lag1'] = df['Returns'].shift(1)
    df['Return_Lag2'] = df['Returns'].shift(2)
    df['Return_Lag3'] = df['Returns'].shift(3)

    # VIX zenginleştirmesi (varsa)
    if 'VIX_Close' not in df.columns:
        df['VIX_Close'] = 20.0
    if 'VIX_MA_5' not in df.columns:
        df['VIX_MA_5'] = df['VIX_Close'].rolling(window=5, min_periods=1).mean()
    else:
        df['VIX_MA_5'] = df['VIX_MA_5'].fillna(df['VIX_Close'])

    # Label: ertesi gün yönü (lookahead yok — shift(-1) label için)
    if include_target:
        df['Target_Return'] = df['Returns'].shift(-1)
        df['Target_Direction'] = (df['Target_Return'] > 0).astype(int)
        return df.dropna()
    else:
        # Inference: feature satırlarında NaN varsa düşür, ama son satırı koru
        feature_cols = [
            'Returns', 'Returns_5d', 'Returns_10d',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_PctB', 'ATR_Pct', 'Volume_Ratio',
            'SMA_Diff', 'Momentum_5', 'Momentum_10',
            'Volatility_20', 'Volatility_5',
            'Return_Lag1', 'Return_Lag2', 'Return_Lag3',
            'VIX_Close', 'VIX_MA_5'
        ]
        return df.dropna(subset=feature_cols)


# ===================================================================
# 3. ML MODELİ — LightGBM / XGBoost / sklearn fallback
# ===================================================================
class MLPredictor:
    """Gradient boosting tabanlı yön tahmin modeli."""

    FEATURE_COLS = [
        'Returns', 'Returns_5d', 'Returns_10d',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_PctB', 'ATR_Pct', 'Volume_Ratio',
        'SMA_Diff', 'Momentum_5', 'Momentum_10',
        'Volatility_20', 'Volatility_5',
        'Return_Lag1', 'Return_Lag2', 'Return_Lag3',
        'VIX_Close', 'VIX_MA_5'
    ]

    def __init__(self):
        self.model = None
        self.model_name = "None"
        self.accuracy = 0.0
        self.feature_importance = {}
        self.is_trained = False
        self.split_idx = 0  # out-of-sample boundary

    def _select_backend(self):
        if _HAS_LIGHTGBM:
            return "lightgbm"
        elif _HAS_XGBOOST:
            return "xgboost"
        elif _HAS_SKLEARN:
            return "sklearn"
        return None

    def train(self, df, test_size=0.3, optimize_hyperparams=True):
        """
        Walk-Forward CV (TimeSeriesSplit) + RandomizedSearchCV ile eğitir.
        İlk %70 train, son %30 test. Hiperparametre optimizasyonu opsiyonel.
        """
        backend = self._select_backend()
        if backend is None:
            return {"error": "ML kütüphanesi bulunamadı. lightgbm, xgboost veya scikit-learn kurun."}

        feature_df = engineer_features(df)
        if len(feature_df) < 100:
            return {"error": f"Yetersiz veri: {len(feature_df)} satır (min 100 gerekli)."}

        split_idx = int(len(feature_df) * (1 - test_size))
        self.split_idx = split_idx
        train_df = feature_df.iloc[:split_idx]
        test_df = feature_df.iloc[split_idx:]

        X_train = train_df[self.FEATURE_COLS]
        y_train = train_df['Target_Direction']
        X_test = test_df[self.FEATURE_COLS]
        y_test = test_df['Target_Direction']

        best_params = {}

        if backend == "lightgbm":
            base_model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
            self.model_name = "LightGBM"
            if optimize_hyperparams and _HAS_SKLEARN and len(X_train) > 200:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'num_leaves': [15, 31, 63],
                    'subsample': [0.7, 0.8, 1.0],
                }
                tscv = TimeSeriesSplit(n_splits=5)
                search = RandomizedSearchCV(
                    base_model, param_grid, n_iter=8, scoring='accuracy',
                    cv=tscv, random_state=42, n_jobs=-1
                )
                search.fit(X_train, y_train)
                self.model = search.best_estimator_
                best_params = search.best_params_
            else:
                self.model = lgb.LGBMClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.05,
                    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbose=-1
                )
                self.model.fit(X_train, y_train)

        elif backend == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, use_label_encoder=False, eval_metric='logloss'
            )
            self.model_name = "XGBoost"
            self.model.fit(X_train, y_train)

        else:
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            self.model_name = "RandomForest"
            if optimize_hyperparams and len(X_train) > 200:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                }
                tscv = TimeSeriesSplit(n_splits=5)
                search = RandomizedSearchCV(
                    base_model, param_grid, n_iter=8, scoring='accuracy',
                    cv=tscv, random_state=42, n_jobs=-1
                )
                search.fit(X_train, y_train)
                self.model = search.best_estimator_
                best_params = search.best_params_
            else:
                self.model = GradientBoostingClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, random_state=42
                )
                self.model_name = "sklearn-GradientBoosting"
                self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.FEATURE_COLS,
                self.model.feature_importances_.tolist()
            ))
        self.is_trained = True

        return {
            "model": self.model_name,
            "accuracy": round(self.accuracy, 4),
            "train_size": len(train_df),
            "test_size": len(test_df),
            "best_params": best_params if best_params else None,
            "feature_importance": self.feature_importance,
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
            if _HAS_SKLEARN else None
        }

    def predict_latest(self, df):
        """En son veri noktası için tahmin üretir (include_target=False ile son satırı korur)."""
        if not self.is_trained:
            return None
        feature_df = engineer_features(df, include_target=False)
        if len(feature_df) == 0:
            return None
        latest = feature_df[self.FEATURE_COLS].iloc[-1:]
        pred = int(self.model.predict(latest)[0])
        proba = float(self.model.predict_proba(latest)[0].max())
        return {"direction": pred, "probability": round(proba, 4)}


# ===================================================================
# 4. RISK YÖNETİMİ
# ===================================================================
class RiskManager:
    """
    Position sizing, stop-loss, take-profit ve max drawdown kontrolü.
    """

    def __init__(self, risk_per_trade=0.02, max_position_pct=0.25,
                 stop_loss_atr_mult=2.0, take_profit_atr_mult=3.0,
                 max_daily_loss_pct=0.05, commission_pct=0.001,
                 slippage_pct=0.0005):
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.take_profit_atr_mult = take_profit_atr_mult
        self.max_daily_loss_pct = max_daily_loss_pct
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def calculate_position_size(self, portfolio_value, entry_price, atr):
        """
        Risk-per-trade yöntemiyle pozisyon büyüklüğü hesaplar.
        Stop-loss = entry - ATR * multiplier
        """
        risk_amount = portfolio_value * self.risk_per_trade
        stop_distance = atr * self.stop_loss_atr_mult
        if stop_distance <= 0 or entry_price <= 0:
            return 0, 0.0, 0.0

        max_shares_by_risk = int(risk_amount / stop_distance)
        max_shares_by_capital = int(
            (portfolio_value * self.max_position_pct) / entry_price
        )
        shares = min(max_shares_by_risk, max_shares_by_capital)

        stop_loss = entry_price - stop_distance
        take_profit = entry_price + (atr * self.take_profit_atr_mult)

        return shares, round(stop_loss, 2), round(take_profit, 2)

    def calculate_costs(self, shares, price):
        """Komisyon + slippage maliyeti."""
        cost = shares * price * (self.commission_pct + self.slippage_pct)
        return cost

    def check_max_drawdown(self, portfolio_value, peak_value):
        """Max drawdown kontrolü — limit aşılırsa işlem durdurulur."""
        if peak_value <= 0:
            return True
        drawdown = (peak_value - portfolio_value) / peak_value
        return drawdown < self.max_daily_loss_pct


# ===================================================================
# 5. BACKTEST SİSTEMİ + PERFORMANS METRİKLERİ
# ===================================================================
class Backtester:
    """
    Walk-forward backtest motoru.
    - Zaman serisi split (lookahead yok)
    - İşlem maliyetleri (komisyon + slippage)
    - Equity curve, trade log
    - Risk yönetimi entegre
    """

    def __init__(self, risk_manager=None):
        self.rm = risk_manager or RiskManager()

    @staticmethod
    def calculate_metrics(returns_series, equity_curve, trades, initial_capital, num_days):
        """
        Sharpe, Sortino, max drawdown, CAGR, win rate, profit factor.
        """
        returns = returns_series.dropna()

        # Sharpe Ratio (yıllık)
        if returns.std() > 0:
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe = 0.0

        # Sortino Ratio (sadece downside volatilite)
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = np.sqrt(252) * returns.mean() / downside.std()
        else:
            sortino = 0.0

        # Max Drawdown (pozitif olarak sakla)
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        max_dd = abs(float(drawdown.min()))

        # CAGR
        if num_days > 0 and initial_capital > 0:
            final = equity_curve.iloc[-1]
            cagr = (final / initial_capital) ** (252 / num_days) - 1
        else:
            cagr = 0.0

        # Win rate
        if trades:
            wins = [t for t in trades if t['pnl'] > 0]
            losses = [t for t in trades if t['pnl'] <= 0]
            win_rate = len(wins) / len(trades) if trades else 0

            gross_profit = sum(t['pnl'] for t in wins)
            gross_loss = abs(sum(t['pnl'] for t in losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            win_rate = 0.0
            profit_factor = 0.0

        # Volatilite (yıllık)
        volatility = returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0

        # Total return
        total_return = (equity_curve.iloc[-1] / initial_capital - 1) * 100

        return {
            'sharpe': round(sharpe, 4),
            'sortino': round(sortino, 4),
            'max_drawdown': round(max_dd * 100, 2),
            'cagr': round(cagr * 100, 2),
            'win_rate': round(win_rate * 100, 2),
            'profit_factor': round(profit_factor, 4),
            'volatility': round(volatility * 100, 2),
            'total_return': round(total_return, 2),
            'total_trades': len(trades),
            'initial_capital': initial_capital,
            'final_capital': round(float(equity_curve.iloc[-1]), 2),
        }

    def run(self, df_1d, kod, initial_capital=10000,
            strategy="ml", ml_predictor=None):
        """
        Backtest çalıştırır.

        strategy: "ml" (ML tahminli) veya "rule" (RSI+Bollinger kuralı)
        df_1d: Günlük OHLCV DataFrame
        """
        df = df_1d.copy()
        if len(df) < 100:
            return {"error": f"Yetersiz veri: {len(df)} satır"}

        # Feature engineering
        feat = engineer_features(df)

        # --- Cash & position tracking (doğru muhasebe) ---
        cash = float(initial_capital)
        peak_equity = float(initial_capital)
        position = 0  # shares held
        entry_price = 0.0
        entry_cost = 0.0  # alışta ödenen komisyon+slippage
        stop_loss = 0.0
        take_profit = 0.0
        highest_price = 0.0  # Trailing stop için zirve takibi
        trailing_stop = 0.0  # ATR tabanlı izleyen stop
        entry_date = None
        trades = []
        equity_dates = []
        equity_values = []

        # Warm-up: ilk 50 gün feature stabilizasyonu için atla
        start_idx = 50

        # ML stratejisi: sadece out-of-sample bölgeden başla
        if strategy == "ml" and ml_predictor and ml_predictor.is_trained:
            ml_start = max(start_idx, ml_predictor.split_idx)
        else:
            ml_start = start_idx

        for i in range(ml_start, len(feat) - 1):
            row = feat.iloc[i]
            next_row = feat.iloc[i + 1]
            date_str = str(row.name) if hasattr(row, 'name') else f"Day_{i}"
            price = float(row['Close'])
            atr = float(row['ATR']) if row['ATR'] > 0 else price * 0.02

            # Mevcut equity
            current_equity = cash + (position * price if position > 0 else 0)
            peak_equity = max(peak_equity, current_equity)

            # Max drawdown kontrolü
            if not self.rm.check_max_drawdown(current_equity, peak_equity):
                if position > 0:
                    exit_cost = self.rm.calculate_costs(position, price)
                    proceeds = position * price - exit_cost
                    cash += proceeds
                    pnl = proceeds - (position * entry_price + entry_cost)
                    trades.append({
                        'kod': kod, 'entry_date': str(entry_date),
                        'entry_price': round(entry_price, 2),
                        'exit_date': date_str, 'exit_price': round(price, 2),
                        'shares': position, 'pnl': round(pnl, 2),
                        'return_pct': round((pnl / (position * entry_price + entry_cost)) * 100, 2),
                        'side': 'LONG', 'reason': 'Max drawdown stop'
                    })
                    position = 0
                equity_dates.append(date_str)
                equity_values.append(cash)
                continue

            # --- Sinyal üretimi ---
            signal = 0  # 1=buy, -1=sell, 0=hold

            if strategy == "ml" and ml_predictor and ml_predictor.is_trained:
                # ML sinyali (out-of-sample)
                feat_row = feat[ml_predictor.FEATURE_COLS].iloc[i:i+1]
                pred = int(ml_predictor.model.predict(feat_row)[0])
                proba = float(ml_predictor.model.predict_proba(feat_row)[0].max())
                signal = 1 if pred == 1 and proba > 0.55 else 0
            else:
                # Rule-based: RSI < 30 + BB_PctB < 0.1 => buy
                #            RSI > 70 + BB_PctB > 0.9 => sell
                rsi_val = float(row['RSI'])
                bb_pctb = float(row['BB_PctB'])
                if rsi_val < 30 and bb_pctb < 0.1:
                    signal = 1
                elif rsi_val > 70 and bb_pctb > 0.9:
                    signal = -1

            # --- İşlem yönetimi ---
            next_price = float(next_row['Close'])

            # Stop-loss / Take-profit / Trailing stop / sell sinyali (mevcut pozisyon varsa)
            if position > 0:
                # Trailing stop güncelleme: fiyat yeni zirve yaptıysa stop'u yukarı çek
                if next_price > highest_price:
                    highest_price = next_price
                    new_trailing = highest_price - (atr * self.rm.stop_loss_atr_mult)
                    if new_trailing > trailing_stop:
                        trailing_stop = new_trailing

                exit_reason = None
                # En yüksek öncelik: trailing stop (daha agresif koruma)
                if next_price <= trailing_stop and trailing_stop > 0:
                    exit_reason = 'Trailing stop'
                elif next_price <= stop_loss:
                    exit_reason = 'Stop-loss'
                elif next_price >= take_profit:
                    exit_reason = 'Take-profit'
                elif signal == -1:
                    exit_reason = 'Signal sell'

                if exit_reason:
                    exit_cost = self.rm.calculate_costs(position, next_price)
                    proceeds = position * next_price - exit_cost
                    cash += proceeds
                    pnl = proceeds - (position * entry_price + entry_cost)
                    trades.append({
                        'kod': kod, 'entry_date': str(entry_date),
                        'entry_price': round(entry_price, 2),
                        'exit_date': str(next_row.name) if hasattr(next_row, 'name') else f"Day_{i+1}",
                        'exit_price': round(next_price, 2),
                        'shares': position, 'pnl': round(pnl, 2),
                        'return_pct': round((pnl / (position * entry_price + entry_cost)) * 100, 2),
                        'side': 'LONG', 'reason': exit_reason
                    })
                    position = 0

            # Yeni alım sinyali (cash'ten hisse al)
            if signal == 1 and position == 0:
                shares, sl, tp = self.rm.calculate_position_size(cash, next_price, atr)
                if shares > 0:
                    buy_cost = self.rm.calculate_costs(shares, next_price)
                    total_cost = shares * next_price + buy_cost
                    if total_cost <= cash:
                        cash -= total_cost
                        position = shares
                        entry_price = next_price
                        entry_cost = buy_cost
                        stop_loss = sl
                        take_profit = tp
                        highest_price = next_price
                        trailing_stop = sl  # Başlangıçta stop = fixed stop, sonra yukarı çıkar
                        entry_date = str(next_row.name) if hasattr(next_row, 'name') else f"Day_{i+1}"

            # Equity (cash + pozisyon değeri)
            equity = cash + (position * next_price if position > 0 else 0)
            equity_dates.append(date_str)
            equity_values.append(equity)
            peak_equity = max(peak_equity, equity)

        # Son pozisyonu kapat
        if position > 0:
            last_price = float(feat.iloc[-1]['Close'])
            exit_cost = self.rm.calculate_costs(position, last_price)
            proceeds = position * last_price - exit_cost
            cash += proceeds
            pnl = proceeds - (position * entry_price + entry_cost)
            trades.append({
                'kod': kod, 'entry_date': str(entry_date),
                'entry_price': round(entry_price, 2),
                'exit_date': str(feat.index[-1]),
                'exit_price': round(last_price, 2),
                'shares': position, 'pnl': round(pnl, 2),
                'return_pct': round((pnl / (position * entry_price + entry_cost)) * 100, 2),
                'side': 'LONG', 'reason': 'Backtest end'
            })
            if equity_values:
                equity_values[-1] = cash
            position = 0

        # Sonuçlar — getiri equity curve'den türetilir
        equity_curve = pd.Series(equity_values, dtype=float)
        returns_series = equity_curve.pct_change().dropna()
        num_days = len(equity_values)

        metrics = self.calculate_metrics(
            returns_series, equity_curve, trades, initial_capital, num_days
        )
        metrics['kod'] = kod
        metrics['equity_curve'] = list(zip(equity_dates, [round(float(v), 2) for v in equity_values]))

        return {
            'metrics': metrics,
            'trades': trades,
            'equity_curve': equity_curve,
            'equity_dates': equity_dates
        }


# ===================================================================
# VERİ ÇEKME — Yahoo Finance (Dual: 15m + 1d)
# ===================================================================
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Connection': 'keep-alive'
})


def fetch_yahoo_data(h_kod, interval="1d", range_param="3mo", include_vix=False):
    """Yahoo Finance'den OHLCV verisi çeker. include_vix=True ise VIX volatilite verisini de ekler."""
    h_clean = str(h_kod).strip().upper()
    h_yf = h_clean if h_clean.endswith(".ST") else f"{h_clean}.ST"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{h_yf}?range={range_param}&interval={interval}"

    try:
        res = session.get(url, timeout=8)
        if res.status_code != 200:
            return None, None
        data = res.json()['chart']['result'][0]
        meta = data.get('meta', {})
        sirket_adi = meta.get('shortName') or meta.get('longName') or h_clean

        timestamps = data.get('timestamp', [])
        if not timestamps:
            return None, None

        quote_data = data['indicators']['quote'][0]
        df = pd.DataFrame({
            'Open': quote_data.get('open'),
            'Close': quote_data.get('close'),
            'High': quote_data.get('high'),
            'Low': quote_data.get('low'),
            'Volume': quote_data.get('volume')
        }, index=pd.to_datetime(timestamps, unit='s')).dropna()

        # VIX (Piyasa Volatilite Endeksi) zenginleştirmesi
        if include_vix and interval == "1d":
            try:
                vix_url = "https://query1.finance.yahoo.com/v8/finance/chart/^VIX?range=" + range_param + "&interval=1d"
                vix_res = session.get(vix_url, timeout=5)
                if vix_res.status_code == 200:
                    vix_data = vix_res.json()['chart']['result'][0]
                    vix_ts = vix_data.get('timestamp', [])
                    vix_close = vix_data['indicators']['quote'][0].get('close')
                    if vix_ts and vix_close:
                        vix_df = pd.DataFrame({'VIX_Close': vix_close}, index=pd.to_datetime(vix_ts, unit='s')).dropna()
                        df = df.join(vix_df, how='left')
                        df['VIX_Close'] = df['VIX_Close'].ffill().bfill()
                    else:
                        df['VIX_Close'] = 20.0
                else:
                    df['VIX_Close'] = 20.0
            except Exception:
                df['VIX_Close'] = 20.0
            df['VIX_MA_5'] = df['VIX_Close'].rolling(window=5).mean()

        return df, sirket_adi
    except Exception:
        return None, None


def tek_hisse_analiz_et(h_kod):
    """İki katmanlı (15m + 1d) analiz yapar."""
    h_clean = str(h_kod).strip().upper()

    df_15m, sirket_adi = fetch_yahoo_data(h_clean, "15m", "5d")
    df_1d, _ = fetch_yahoo_data(h_clean, "1d", "3mo")

    if df_15m is None or df_1d is None:
        return None
    if len(df_15m) < 30 or len(df_1d) < 30:
        return None

    # --- 15 Dakikalık Veri ---
    son_fiyat = float(df_15m['Close'].iloc[-1])

    sma_20 = df_15m['Close'].rolling(window=20).mean()
    std_20 = df_15m['Close'].rolling(window=20).std()
    upper_band = (sma_20 + (std_20 * 2)).iloc[-1]
    lower_band = (sma_20 - (std_20 * 2)).iloc[-1]

    vol_sum = df_15m['Volume'].rolling(14).sum()
    vwap = (df_15m['Close'] * df_15m['Volume']).rolling(14).sum() / vol_sum
    son_vwap = round(vwap.iloc[-1], 2)

    son_rsi_15m = float(compute_rsi(df_15m['Close'], 14).iloc[-1])

    # --- Günlük Veri ---
    close_1d = df_1d['Close']
    atr_14_1d = float(compute_atr(df_1d, 14).iloc[-1])
    atr_yuzde = (atr_14_1d / son_fiyat) * 100

    son_rsi_1d = float(compute_rsi(close_1d, 14).iloc[-1])

    gunluk_getiri = close_1d.pct_change().dropna()
    ema_getiri = gunluk_getiri.ewm(span=10, adjust=False).mean().iloc[-1] * 100

    if son_rsi_1d > 75:
        momentum = -0.15
    elif son_rsi_1d < 30:
        momentum = 0.15
    else:
        momentum = (son_rsi_1d - 50) / 100.0

    vol_s_1d = df_1d['Volume'].tail(10)
    hacim_kati = float(vol_s_1d.iloc[-1] / vol_s_1d.mean()) if vol_s_1d.mean() > 0 else 1.0
    hacim_etkisi = min(hacim_kati, 2.0)

    # --- Tahminler ---
    tahminler_yuzde = {}
    kumbulatif_fiyat = son_fiyat
    aktif_momentum = momentum

    for gun in range(1, 4):
        gun_artis_yuzde = ema_getiri + (aktif_momentum * atr_yuzde * hacim_etkisi)
        gun_artis_yuzde = max(min(gun_artis_yuzde, atr_yuzde * 1.5), -atr_yuzde * 1.5)
        kumbulatif_fiyat = kumbulatif_fiyat * (1 + (gun_artis_yuzde / 100))
        tahminler_yuzde[f'{gun}. Gün Tahmin (%)'] = round(
            ((kumbulatif_fiyat - son_fiyat) / son_fiyat) * 100, 2)
        aktif_momentum *= 0.5
        hacim_etkisi = max(1.0, hacim_etkisi * 0.8)

    yarin_alis = round(son_fiyat - (atr_14_1d * 0.35), 2)
    potansiyel_pik = round(upper_band + (atr_14_1d * 0.1), 2)

    # --- Sinyal ---
    if son_fiyat >= upper_band and son_rsi_15m > 70:
        sinyal = "PİK YAPTI (SAT)"
        strateji_metni = "Fiyat Bollinger üst bandını aştı. Geri çekilme riski yüksek."
        saatlik_yon = -1.0
    elif hacim_kati > 1.2 and son_fiyat > son_vwap and tahminler_yuzde['1. Gün Tahmin (%)'] > 0:
        sinyal = "GÜÇLÜ AL & TUT"
        strateji_metni = f"Fiyat VWAP üzerinde ve momentum yüksek. Pik hedefi {potansiyel_pik} SEK."
        saatlik_yon = 1.5
    elif son_fiyat < lower_band and son_rsi_15m < 30:
        sinyal = "DİPTEN TEPKİ (AL)"
        strateji_metni = f"Hisse gün içi aşırı satıldı. {son_vwap} SEK seviyesine kadar tepki sıçraması beklenebilir."
        saatlik_yon = 1.0
    else:
        sinyal = "NÖTR (İZLE)"
        strateji_metni = "Net bir trend yok, yatay seyir hakim."
        saatlik_yon = 0.0

    beklenen_1h_getiri = round((atr_14_1d / son_fiyat / 4) * 100 * saatlik_yon, 2)

    result = {
        'Şirket Adı': sirket_adi,
        'Kod': h_clean,
        'Son Fiyat (SEK)': round(son_fiyat, 2),
        '1 Saatlik Yön (%)': beklenen_1h_getiri,
        '1. Gün Tahmin (%)': tahminler_yuzde['1. Gün Tahmin (%)'],
        '2. Gün Tahmin (%)': tahminler_yuzde['2. Gün Tahmin (%)'],
        '3. Gün Tahmin (%)': tahminler_yuzde['3. Gün Tahmin (%)'],
        'Sinyal': sinyal,
        'RSI (15m)': round(son_rsi_15m, 1),
        'VWAP': son_vwap,
        'Potansiyel Pik': potansiyel_pik,
        'Yarınki Alış': yarin_alis,
        'Strateji': strateji_metni,
        'Analiz Zamanı': datetime.now().strftime("%Y-%m-%d %H:%M")
    }

    # Not: SQLite kaydı ThreadPoolExecutor dışında sıralı yapılır
    # (paralel yazım kilitleme sorununa yol açar)

    return result


# ===================================================================
# SIDEBAR — Hisse Listesi & Ayarlar
# ===================================================================
st.sidebar.markdown("## ⚙️ Kontrol Paneli")

varsayilan_liste = (
    "ELUX-B, ANOD-B, LIME, ELUX-A, MIPS, ARLA, NTEK-B, PRIC-B, BILL, BOOZT, NANO, QLINEA, XBRANE, HOLM-B, "
    "PRFO, RAY-B, BESQ-B, PROFF, CLAS-B, REJL-B, SEZI, SECT-B, BIOA-B, SKIS-B, RROS, CER, YUBICO, HOLM-A, "
    "KABE-B, CARA, ESSITY-A, SOBI, EPEN, ESSITY-B, PREV-B, CEVI, HMS, HEXA-B, SWEC-A, THULE, APOTEA, TROAX, "
    "VIMIAN, NEWA-B, CTT, SVT, BORG, LIFCO-B, SLEEP, SAND, TREL-B, MSAB-B, GARO, MCOV-B, BIOG-B, BONES, "
    "PCELL, SINT, SWEC-B, EPRO-B, HEM, ASSA-B, INWI, VITR, NIL-B, NOLA-B, BEIJ-B, FOI-B, MCAP, SKF-B, "
    "CAMX, G5EN, BEIJ-REF-B, SKF-A, MMGR-B, GETI-B, CCC, BACT-B, CINT, AAK, ALLEI, INVI, DUNI, CTEK, EWRK, "
    "ADDB-B, PION-B, NIBE-B, LAGR-B, NETI-B, HPOL-B, EPI-A, NELLY, FMM-B, CRAD-B, SYST, INDT, NOTE, XANO-B, "
    "ATCO-A, ENGCON-B, EPI-B, ALFA, ATCO-B, BUFAB, AQ, DYNA, MEKE-B, PREC, IRLAB-A, MYCR, DEDI-B, WISE"
)

girdi = st.sidebar.text_area("📋 Hisse Listeniz:", value=varsayilan_liste, height=300)
hisseler = [h.strip() for h in girdi.split(",") if h.strip()]

# Hisse adlarını paralel çek ve cache'le
@st.cache_data(ttl=3600, show_spinner=False)
def get_ticker_name_map(ticker_tuple):
    """Yahoo Finance'den hisse adlarını paralel olarak çeker."""
    tickers = list(ticker_tuple)
    name_map = {}
    def _fetch(t):
        h_clean = str(t).strip().upper()
        h_yf = h_clean if h_clean.endswith(".ST") else f"{h_clean}.ST"
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{h_yf}?range=1d&interval=1d"
        try:
            res = session.get(url, timeout=5)
            if res.status_code == 200:
                meta = res.json()['chart']['result'][0].get('meta', {})
                name = meta.get('shortName') or meta.get('longName') or h_clean
                return t, name
        except Exception:
            pass
        return t, h_clean
    with ThreadPoolExecutor(max_workers=20) as executor:
        for future in as_completed([executor.submit(_fetch, t) for t in tickers]):
            t, name = future.result()
            name_map[t] = name
    return name_map

ticker_names = {}
if hisseler:
    with st.spinner("Hisse adları yükleniyor..."):
        ticker_names = get_ticker_name_map(tuple(hisseler))

def format_hisse(ticker):
    """Dropdown'da tam şirket adını göster, ticker'ı değer olarak kullan."""
    return ticker_names.get(ticker, ticker)

# Hisse sayacı
st.sidebar.caption(f"📊 Listedeki hisse sayısı: **{len(hisseler)}**")

# ML backend info
ml_backend = "Yok"
if _HAS_LIGHTGBM:
    ml_backend = "LightGBM"
elif _HAS_XGBOOST:
    ml_backend = "XGBoost"
elif _HAS_SKLEARN:
    ml_backend = "sklearn-GradientBoosting"

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Sistem Durumu")
st.sidebar.markdown(f"- **ML Backend:** `{ml_backend}`")
st.sidebar.markdown(f"- **Database:** `SQLite` ✓")
st.sidebar.markdown(f"- **Veri Kaynağı:** `Yahoo Finance` ✓")

if ml_backend == "Yok":
    st.sidebar.warning("ML için: `pip install lightgbm xgboost scikit-learn`")

st.sidebar.markdown("---")
st.sidebar.caption("⚠️ Eğitim amaçlıdır. Yatırım tavsiyesi değildir.")


# ===================================================================
# BAŞLIK
# ===================================================================
st.markdown("# ⚡ Borsa Panel")
st.markdown("*15m Zirve Avcısı · ML Tahmin · Backtest · Risk Yönetimi · Sharpe Metrikleri*")
st.markdown("---")


# ===================================================================
# TABS
# ===================================================================
tab_analiz, tab_backtest, tab_ml, tab_risk, tab_gecmis = st.tabs([
    "📊 Canlı Analiz",
    "🔬 Backtest",
    "🤖 ML Tahmin",
    "🛡️ Risk Yönetimi",
    "📜 Geçmiş"
])


# ===================================================================
# TAB 1: CANLI ANALİZ
# ===================================================================
with tab_analiz:
    gorunum_modu = st.radio(
        "Görünüm Modu:",
        ["Kart Görünümü", "Tablo Görünümü"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if "pro_analiz_df" not in st.session_state:
        if os.path.exists(DATA_FILE):
            st.session_state.pro_analiz_df = pd.read_csv(DATA_FILE)
        else:
            st.session_state.pro_analiz_df = None

    col_scan1, col_scan2 = st.columns([4, 1])
    with col_scan1:
        st.caption(f"📊 Taranacak hisse sayısı: **{len(hisseler)}**")
    with col_scan2:
        pass

    if st.button("🔄 Hibrid Analizi Başlat (Day + Swing Trade)", type="primary", use_container_width=True):
        if hisseler:
            rapor = []
            bar = st.progress(0)
            durum = st.empty()
            durum.text("⚡ Hisseler taranıyor... Çift Katmanlı Veri İşleniyor...")

            completed_count = 0
            with ThreadPoolExecutor(max_workers=20) as executor:
                future_to_stock = {executor.submit(tek_hisse_analiz_et, h): h for h in hisseler}
                for future in as_completed(future_to_stock):
                    res = future.result()
                    if res:
                        rapor.append(res)
                    completed_count += 1
                    bar.progress(completed_count / len(hisseler))

            durum.empty()
            bar.empty()

            if rapor:
                df_res = pd.DataFrame(rapor)
                df_res = df_res.sort_values(
                    by=['1. Gün Tahmin (%)', '1 Saatlik Yön (%)'],
                    ascending=[False, False]
                )
                st.session_state.pro_analiz_df = df_res
                df_res.to_csv(DATA_FILE, index=False)

                # SQLite'a sıralı kaydet (paralel kilitleme yok)
                for row_dict in rapor:
                    try:
                        db.save_prediction(row_dict)
                    except Exception:
                        pass

    if st.session_state.get('pro_analiz_df') is not None:
        df_res = st.session_state.pro_analiz_df

        son_zaman = df_res['Analiz Zamanı'].iloc[0] if 'Analiz Zamanı' in df_res.columns else "Bilinmiyor"
        st.info(f"🕒 Son güncelleme: **{son_zaman}**")

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Analiz Edilen", len(df_res))
        c2.metric("Gün İçi İvmeli", len(df_res[df_res['1 Saatlik Yön (%)'] > 0]))
        c3.metric("Yarın Yükseliş Beklenen", len(df_res[df_res['1. Gün Tahmin (%)'] > 0]))

        if gorunum_modu == "Kart Görünümü":
            st.markdown("---")
            st.markdown("### 🔥 Canlı Aksiyon Planları & Çoklu Tahminler")
            for _, row in df_res.iterrows():
                with st.container():
                    # 1) Sinyal/Tavsiye en üstte — kompakt rozet
                    if "AL" in row['Sinyal']:
                        badge_cls = "signal-buy"
                        badge_emoji = "✅"
                    elif "SAT" in row['Sinyal']:
                        badge_cls = "signal-sell"
                        badge_emoji = "❌"
                    else:
                        badge_cls = "signal-hold"
                        badge_emoji = "⏸️"
                    st.markdown(
                        f"<span class='signal-badge {badge_cls}'>"
                        f"{badge_emoji} {row['Sinyal']}</span>",
                        unsafe_allow_html=True
                    )

                    # 2) Şirket adı + fiyat bilgisi tek satırda
                    st.markdown(
                        f"**{row['Şirket Adı']}** "
                        f"<span style='font-size:0.75rem; color:gray;'>({row['Kod']})</span> &nbsp;"
                        f"<span style='font-size:0.8rem;'>"
                        f"Fiyat: {row['Son Fiyat (SEK)']} SEK | "
                        f"RSI: {row['RSI (15m)']} | "
                        f"VWAP: {row['VWAP']} SEK | "
                        f"Pik: {row['Potansiyel Pik']} SEK"
                        f"</span>",
                        unsafe_allow_html=True
                    )

                    # 3) Tahmin değerleri altta — kompakt metric'ler
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("1 Saat", f"%{row['1 Saatlik Yön (%)']:+.2f}")
                    m2.metric("1. Gün", f"%{row['1. Gün Tahmin (%)']:+.2f}")
                    m3.metric("2. Gün", f"%{row['2. Gün Tahmin (%)']:+.2f}")
                    m4.metric("3. Gün", f"%{row['3. Gün Tahmin (%)']:+.2f}")

                    # 4) Strateji + alış tavsiyesi — okunabilir boyutta
                    st.markdown(
                        f"🎯 **Strateji:** {row['Strateji']}"
                    )
                    st.markdown(
                        f"📅 **Alış Tavsiyesi:** Olası bir sabah esnemesinde "
                        f"**{row['Yarınki Alış']} SEK** seviyesi alım için takip edilebilir."
                    )
                    st.markdown("---")
        else:
            st.dataframe(df_res.drop(columns=['Strateji']), use_container_width=True)


# ===================================================================
# TAB 2: BACKTEST
# ===================================================================
with tab_backtest:
    st.markdown("### 🔬 Backtest Sistemi")
    st.caption("Zaman serisi split · Lookahead yok · İşlem maliyetleri · Walk-forward")

    col_bt1, col_bt2, col_bt3 = st.columns(3)
    with col_bt1:
        bt_hisse = st.selectbox(
            "Hisse Seç:",
            options=hisseler if hisseler else ["ELUX-B"],
            index=0,
            format_func=format_hisse,
            key="bt_hisse"
        )
    with col_bt2:
        bt_capital = st.number_input("Başlangıç Sermayesi (SEK):", value=10000, min_value=1000, step=1000, key="bt_capital")
    with col_bt3:
        bt_strategy = st.selectbox(
            "Strateji:",
            ["rule", "ml"],
            format_func=lambda x: "📊 Kural Bazlı (RSI+Bollinger)" if x == "rule" else "🤖 ML Tahminli",
            key="bt_strategy"
        )

    col_bt4, col_bt5 = st.columns(2)
    with col_bt4:
        bt_range = st.selectbox(
            "Veri Aralığı:",
            ["6mo", "1y", "2y", "5y"],
            format_func=lambda x: {"6mo": "6 Ay", "1y": "1 Yıl", "2y": "2 Yıl", "5y": "5 Yıl"}[x],
            index=1,
            key="bt_range"
        )
    with col_bt5:
        bt_risk = st.slider("Risk / İşlem (%):", 1, 5, 2, key="bt_risk")

    st.markdown("")

    if st.button("🚀 Backtest Çalıştır", type="primary", use_container_width=True, key="bt_run"):
        with st.spinner("Veri çekiliyor..."):
            df_bt, bt_name = fetch_yahoo_data(bt_hisse, "1d", bt_range, include_vix=True)

        if df_bt is None or len(df_bt) < 100:
            st.error(f"Yetersiz veri veya hisse bulunamadı: {bt_hisse}")
        else:
            st.info(f"📈 **{bt_hisse}** — {len(df_bt)} gün veri yüklendi")

            # ML predictor (strategi ML ise)
            ml_pred = None
            if bt_strategy == "ml":
                ml_pred = MLPredictor()
                with st.spinner("ML modeli eğitiliyor..."):
                    train_result = ml_pred.train(df_bt)
                if "error" in train_result:
                    st.warning(f"ML hatası: {train_result['error']} — Kural bazlı stratejiye geçiliyor.")
                    bt_strategy = "rule"
                    ml_pred = None
                else:
                    st.success(
                        f"✅ {ml_pred.model_name} eğitildi | "
                        f"Accuracy: %{train_result['accuracy']*100:.1f} | "
                        f"Train: {train_result['train_size']} | Test: {train_result['test_size']}"
                    )

            # Risk manager
            rm = RiskManager(risk_per_trade=bt_risk / 100)

            # Backtest
            bt = Backtester(risk_manager=rm)
            with st.spinner("Backtest çalıştırılıyor..."):
                result = bt.run(df_bt, bt_hisse, initial_capital=bt_capital,
                               strategy=bt_strategy, ml_predictor=ml_pred)

            if "error" in result:
                st.error(result["error"])
            else:
                metrics = result['metrics']

                # SQLite'a kaydet
                run_id = db.save_backtest_run(metrics)
                for trade in result['trades']:
                    db.save_backtest_trade(run_id, trade)

                # Metrik kartları
                st.markdown("---")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("💰 Toplam Getiri", f"%{metrics['total_return']:+.2f}")
                mc2.metric("📊 Sharpe Ratio", f"{metrics['sharpe']:.3f}")
                mc3.metric("📉 Max Drawdown", f"%{metrics['max_drawdown']:.2f}")
                mc4.metric("📈 CAGR", f"%{metrics['cagr']:.2f}")

                mc5, mc6, mc7, mc8 = st.columns(4)
                mc5.metric("🎯 Win Rate", f"%{metrics['win_rate']:.1f}")
                mc6.metric("⚖️ Sortino Ratio", f"{metrics['sortino']:.3f}")
                mc7.metric("💵 Profit Factor", f"{metrics['profit_factor']:.2f}")
                mc8.metric("📊 Volatilite (Yıllık)", f"%{metrics['volatility']:.2f}")

                st.markdown("---")

                # Equity curve
                st.markdown("### 📊 Equity Curve")
                eq_dates, eq_values = zip(*metrics['equity_curve'])
                eq_df = pd.DataFrame({'Tarih': eq_dates, 'Portfolio (SEK)': eq_values})
                st.line_chart(eq_df.set_index('Tarih'), use_container_width=True)

                col_fin1, col_fin2 = st.columns(2)
                with col_fin1:
                    st.metric("Başlangıç Sermayesi", f"{metrics['initial_capital']:,.0f} SEK")
                with col_fin2:
                    st.metric("Final Sermaye", f"{metrics['final_capital']:,.0f} SEK")

                # Trade log
                if result['trades']:
                    st.markdown("---")
                    st.markdown(f"### 📋 İşlem Geçmişi ({len(result['trades'])} işlem)")
                    trades_df = pd.DataFrame(result['trades'])
                    st.dataframe(trades_df[['entry_date', 'entry_price', 'exit_date',
                                           'exit_price', 'shares', 'pnl',
                                           'return_pct', 'reason']], use_container_width=True)


# ===================================================================
# TAB 3: ML TAHMIN
# ===================================================================
with tab_ml:
    st.markdown("### 🤖 ML Yön Tahmin Modeli")
    st.caption("Gradient Boosting · 18 Feature · Zaman Serisi Split · Lookahead Yok")

    if ml_backend == "Yok":
        st.error("ML kütüphanesi kurulu değil. `pip install lightgbm xgboost scikit-learn`")
    else:
        col_ml1, col_ml2 = st.columns(2)
        with col_ml1:
            ml_hisse = st.selectbox(
                "Hisse Seç:",
                options=hisseler if hisseler else ["ELUX-B"],
                index=0,
                format_func=format_hisse,
                key="ml_hisse"
            )
        with col_ml2:
            ml_range = st.selectbox(
                "Veri Aralığı:",
                ["6mo", "1y", "2y", "5y"],
                format_func=lambda x: {"6mo": "6 Ay", "1y": "1 Yıl", "2y": "2 Yıl", "5y": "5 Yıl"}[x],
                index=2,
                key="ml_range"
            )

        st.markdown("")

        if st.button("🎯 Model Eğit & Tahmin Et", type="primary", use_container_width=True, key="ml_train"):
            with st.spinner("Veri çekiliyor..."):
                df_ml, ml_name = fetch_yahoo_data(ml_hisse, "1d", ml_range, include_vix=True)

            if df_ml is None or len(df_ml) < 100:
                st.error(f"Yetersiz veri: {ml_hisse}")
            else:
                predictor = MLPredictor()
                with st.spinner("Model eğitiliyor..."):
                    result = predictor.train(df_ml)

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(
                        f"✅ {result['model']} | Accuracy: %{result['accuracy']*100:.1f}"
                    )

                    # Hiperparametre optimizasyonu sonucu
                    if result.get('best_params'):
                        st.caption(f"🔧 **En İyi Parametreler:** {result['best_params']}")
                    st.caption("📊 Walk-Forward CV (TimeSeriesSplit) + RandomizedSearchCV ile eğitildi")

                    # Son tahmin
                    pred = predictor.predict_latest(df_ml)
                    if pred:
                        yon = "📈 YÜKSELİŞ" if pred['direction'] == 1 else "📉 DÜŞÜŞ"
                        st.info(
                            f"**{ml_hisse}** son tahmin: {yon} | "
                            f"Güven: %{pred['probability']*100:.1f}"
                        )

                        # SQLite'a kaydet
                        db.save_ml_prediction(
                            ml_hisse, pred['direction'], pred['probability'],
                            result['model'], result['accuracy'],
                            result.get('feature_importance', {})
                        )

                    # Feature importance
                    if result.get('feature_importance'):
                        st.markdown("---")
                        st.markdown("### 🔧 Feature Importance")
                        fi_df = pd.DataFrame([
                            {'Feature': k, 'Importance': v}
                            for k, v in sorted(
                                result['feature_importance'].items(),
                                key=lambda x: x[1], reverse=True
                            )
                        ])
                        st.bar_chart(fi_df.set_index('Feature'), use_container_width=True)

                    # Classification report
                    if result.get('classification_report'):
                        st.markdown("---")
                        st.markdown("### 📋 Sınıflandırma Raporu")
                        cr = result['classification_report']
                        cr_df = pd.DataFrame(cr).T.drop(columns=['support'], errors='ignore')
                        st.dataframe(cr_df, use_container_width=True)

                    # Feature açıklamaları
                    st.markdown("---")
                    with st.expander("📝 Kullanılan Feature'lar (20 adet)"):
                        feat_desc = [
                            ("Returns", "Günlük getiri oranı"),
                            ("Returns_5d", "5 günlük kümülatif getiri"),
                            ("Returns_10d", "10 günlük kümülatif getiri"),
                            ("RSI", "Relative Strength Index (14)"),
                            ("MACD", "MACD hattı (12,26)"),
                            ("MACD_Signal", "MACD sinyal hattı (9)"),
                            ("MACD_Hist", "MACD histogramı"),
                            ("BB_PctB", "Bollinger %B pozisyonu"),
                            ("ATR_Pct", "ATR / Fiyat (%)"),
                            ("Volume_Ratio", "Hacim / 20 günlük ortalama hacim"),
                            ("SMA_Diff", "(SMA20 - SMA50) / SMA50"),
                            ("Momentum_5", "5 günlük momentum"),
                            ("Momentum_10", "10 günlük momentum"),
                            ("Volatility_20", "20 günlük getiri volatilitesi"),
                            ("Volatility_5", "5 günlük getiri volatilitesi"),
                            ("Return_Lag1", "1 gün önceki getiri"),
                            ("Return_Lag2", "2 gün önceki getiri"),
                            ("Return_Lag3", "3 gün önceki getiri"),
                            ("VIX_Close", "VIX volatilite endeksi (piyasa korkusu)"),
                            ("VIX_MA_5", "VIX 5 günlük hareketli ortalaması"),
                        ]
                        for name, desc in feat_desc:
                            st.write(f"- **{name}**: {desc}")


# ===================================================================
# TAB 4: RISK YÖNETİMİ
# ===================================================================
with tab_risk:
    st.markdown("### 🛡️ Risk Yönetimi Hesaplayıcı")
    st.caption("Position sizing · Stop-loss · Take-profit · Max drawdown")

    col_rk_hisse, col_rk_fetch = st.columns([3, 1])
    with col_rk_hisse:
        rk_hisse = st.selectbox(
            "Hisse Seç:",
            options=hisseler if hisseler else ["ELUX-B"],
            index=0,
            format_func=format_hisse,
            key="rk_hisse"
        )
    with col_rk_fetch:
        st.write("")
        fetch_clicked = st.button("📥 Veri Çek", key="rk_fetch_btn", use_container_width=True)

    # Veri çekme (buton veya hisse değişince)
    if fetch_clicked or rk_hisse:
        with st.spinner(f"{rk_hisse} verisi çekiliyor..."):
            df_rk, rk_name = fetch_yahoo_data(rk_hisse, "1d", "3mo")

        if df_rk is not None and len(df_rk) >= 50:
            rk_son_fiyat = round(float(df_rk['Close'].iloc[-1]), 2)
            rk_atr_val = round(float(compute_atr(df_rk, 14).iloc[-1]), 2)
            rk_rsi_val = round(float(compute_rsi(df_rk['Close'], 14).iloc[-1]), 1)
            st.success(f"✅ {rk_hisse} — Fiyat: {rk_son_fiyat} SEK | ATR: {rk_atr_val} | RSI: {rk_rsi_val}")
        else:
            rk_son_fiyat = 100.0
            rk_atr_val = 3.0
            rk_rsi_val = 50.0
            if fetch_clicked:
                st.warning("Veri çekilemedi, manuel giriş kullanın.")
    else:
        rk_son_fiyat = 100.0
        rk_atr_val = 3.0
        rk_rsi_val = 50.0

    col_rk1, col_rk2 = st.columns(2)
    with col_rk1:
        rk_portfolio = st.number_input("Portföy Değeri (SEK):", value=100000, min_value=1000, step=5000, key="rk_portfolio")
        rk_price = st.number_input("Giriş Fiyatı (SEK):", value=float(rk_son_fiyat), min_value=0.1, step=1.0, key="rk_price")
    with col_rk2:
        rk_risk = st.slider("Risk / İşlem (%):", 1, 5, 2, key="rk_risk")
        rk_atr = st.number_input("ATR (SEK):", value=float(rk_atr_val), min_value=0.01, step=0.1, key="rk_atr")

    col_rk3, col_rk4 = st.columns(2)
    with col_rk3:
        rk_sl_mult = st.slider("Stop-Loss (ATR x):", 1.0, 4.0, 2.0, 0.5, key="rk_sl_mult")
    with col_rk4:
        rk_tp_mult = st.slider("Take-Profit (ATR x):", 1.0, 6.0, 3.0, 0.5, key="rk_tp_mult")

    st.markdown("")

    if st.button("🛡️ Pozisyon Hesapla", type="primary", use_container_width=True, key="rk_calc"):
        rm = RiskManager(
            risk_per_trade=rk_risk / 100,
            stop_loss_atr_mult=rk_sl_mult,
            take_profit_atr_mult=rk_tp_mult
        )
        shares, sl, tp = rm.calculate_position_size(rk_portfolio, rk_price, rk_atr)

        if shares == 0:
            st.warning("Pozisyon açılamaz — risk/fiyat oranı uygun değil.")
        else:
            position_value = shares * rk_price
            risk_amount = rk_portfolio * (rk_risk / 100)
            cost = rm.calculate_costs(shares, rk_price)

            # Risk/reward oranı
            risk_per_share = rk_price - sl
            reward_per_share = tp - rk_price
            rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0

            st.markdown("---")
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("📦 Hisse Adedi", f"{shares:,}")
            rc2.metric("💵 Pozisyon Değeri", f"{position_value:,.0f} SEK")
            rc3.metric("⛔ Stop-Loss", f"{sl:.2f} SEK")
            rc4.metric("🎯 Take-Profit", f"{tp:.2f} SEK")

            rc5, rc6, rc7, rc8 = st.columns(4)
            rc5.metric("💸 Risk Tutarı", f"{risk_amount:,.0f} SEK")
            rc6.metric("📊 Pozisyon / Portföy", f"%{position_value/rk_portfolio*100:.1f}")
            rc7.metric("⚖️ Risk/Ödül", f"1:{rr_ratio:.1f}")
            rc8.metric("💸 İşlem Maliyeti", f"{cost:.2f} SEK")

            st.markdown("---")
            st.info(
                f"📋 **Özet:** {shares:,} hisse @ {rk_price} SEK | "
                f"Stop: {sl:.2f} | Target: {tp:.2f} | "
                f"Risk: {risk_amount:,.0f} SEK ({rk_risk}%) | "
                f"R/R: 1:{rr_ratio:.1f}"
            )


# ===================================================================
# TAB 5: GECMIS (SQLite)
# ===================================================================
with tab_gecmis:
    st.markdown("### 📜 Geçmiş Kayıtlar")
    st.caption("SQLite veritabanından — tahminler, backtest sonuçları ve ML modelleri")

    gecis_sekme = st.radio(
        "Kayıt Tipi:",
        ["Tahminler", "Backtest Sonuçları", "ML Tahminleri"],
        horizontal=True
    )

    if gecis_sekme == "Tahminler":
        col_g1, col_g2 = st.columns([3, 1])
        with col_g2:
            filter_kod_pred = st.selectbox(
                "Hisse Filtresi:",
                ["Tümü"] + hisseler[:20],
                key="filter_pred"
            )
        df_pred = db.get_prediction_history(
            kod=None if filter_kod_pred == "Tümü" else filter_kod_pred,
            limit=100
        )
        if df_pred.empty:
            st.info("Henüz tahmin kaydı yok.")
        else:
            display_cols = ['tarih', 'kod', 'sirket_adi', 'son_fiyat', 'sinyal',
                            'tahmin_1g', 'tahmin_2g', 'tahmin_3g',
                            'rsi_15m', 'vwap', 'analiz_zamani']
            available_cols = [c for c in display_cols if c in df_pred.columns]
            st.dataframe(df_pred[available_cols], use_container_width=True)
            st.download_button(
                "📥 CSV İndir",
                df_pred.to_csv(index=False),
                file_name=f"tahminler_{datetime.now().strftime('%Y%m%d')}.csv"
            )

    elif gecis_sekme == "Backtest Sonuçları":
        df_bt = db.get_backtest_history(limit=50)
        if df_bt.empty:
            st.info("Henüz backtest kaydı yok.")
        else:
            display_cols = ['run_date', 'kod', 'initial_capital', 'final_capital',
                           'total_return', 'sharpe', 'sortino', 'max_drawdown',
                           'cagr', 'win_rate', 'total_trades', 'profit_factor',
                           'volatility']
            available_cols = [c for c in display_cols if c in df_bt.columns]
            st.dataframe(df_bt[available_cols], use_container_width=True)

            # Backtest detayı
            st.markdown("---")
            st.markdown("### 🔍 Backtest Detayı")
            selected_run = st.selectbox(
                "Run seç:",
                df_bt['id'].tolist(),
                format_func=lambda x: f"Run #{x} — {df_bt[df_bt['id']==x]['kod'].iloc[0]} ({df_bt[df_bt['id']==x]['run_date'].iloc[0]})"
            )
            if selected_run:
                trades_df = db.get_trades_for_run(selected_run)
                if not trades_df.empty:
                    st.dataframe(trades_df, use_container_width=True)
                else:
                    st.info("Bu run için işlem kaydı yok.")

    elif gecis_sekme == "ML Tahminleri":
        df_ml = db.get_ml_history(limit=50)
        if df_ml.empty:
            st.info("Henüz ML tahmin kaydı yok.")
        else:
            display_cols = ['tarih', 'kod', 'predicted_direction', 'probability',
                           'model_name', 'accuracy']
            available_cols = [c for c in display_cols if c in df_ml.columns]
            st.dataframe(df_ml[available_cols], use_container_width=True)

            # Feature importance detayı
            if not df_ml.empty:
                latest = df_ml.iloc[0]
                if latest.get('feature_importance'):
                    fi = json.loads(latest['feature_importance'])
                    if fi:
                        st.markdown("---")
                        st.markdown(f"### 🔧 Son Model Feature Importance — {latest['model_name']}")
                        fi_df = pd.DataFrame([
                            {'Feature': k, 'Importance': v}
                            for k, v in sorted(fi.items(), key=lambda x: x[1], reverse=True)
                        ])
                        st.bar_chart(fi_df.set_index('Feature'), use_container_width=True)
