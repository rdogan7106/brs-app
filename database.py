import requests
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from indicators import compute_rsi, compute_atr

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0'})

def fetch_yahoo_data(h_kod, interval="1d", range_param="3mo", include_vix=False):
    h_clean = str(h_kod).strip().upper()
    h_yf = h_clean if h_clean.endswith(".ST") else f"{h_clean}.ST"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{h_yf}?range={range_param}&interval={interval}"
    try:
        res = session.get(url, timeout=8)
        if res.status_code != 200: return None, None
        data = res.json()['chart']['result'][0]
        meta = data.get('meta', {})
        sirket_adi = meta.get('shortName') or meta.get('longName') or h_clean
        timestamps = data.get('timestamp', [])
        if not timestamps: return None, None
        quote_data = data['indicators']['quote'][0]
        df = pd.DataFrame({'Open': quote_data.get('open'), 'Close': quote_data.get('close'), 'High': quote_data.get('high'), 'Low': quote_data.get('low'), 'Volume': quote_data.get('volume')}, index=pd.to_datetime(timestamps, unit='s')).dropna()
        
        if include_vix and interval == "1d":
            try:
                vix_res = session.get(f"https://query1.finance.yahoo.com/v8/finance/chart/^VIX?range={range_param}&interval=1d", timeout=5)
                if vix_res.status_code == 200:
                    vix_data = vix_res.json()['chart']['result'][0]
                    vix_ts, vix_close = vix_data.get('timestamp', []), vix_data['indicators']['quote'][0].get('close')
                    if vix_ts and vix_close:
                        vix_df = pd.DataFrame({'VIX_Close': vix_close}, index=pd.to_datetime(vix_ts, unit='s')).dropna()
                        df = df.join(vix_df, how='left')
                        df['VIX_Close'] = df['VIX_Close'].ffill().bfill()
                    else: df['VIX_Close'] = 20.0
                else: df['VIX_Close'] = 20.0
            except: df['VIX_Close'] = 20.0
            df['VIX_MA_5'] = df['VIX_Close'].rolling(window=5).mean()
            
        return df, sirket_adi
    except: return None, None

def ml_tahmin_ve_backtest(df_15m):
    """XGBoost ile 1 saat (4 mum) sonrasının yönünü tahmin eder ve backtest başarısını hesaplar."""
    df = df_15m.copy()
    
    # Feature Engineering (Özellik Oluşturma)
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['ATR'] = compute_atr(df, 14)
    df['Returns'] = df['Close'].pct_change()
    df['Vol_Change'] = df['Volume'].pct_change()
    
    # 4 mum (1 saat) sonraki fiyat artış durumu (1: Yükseliş, 0: Düşüş/Yatay)
    df['Target'] = (df['Close'].shift(-4) > df['Close']).astype(int)
    
    df_clean = df.dropna()
    if len(df_clean) < 40:
        return 0.5, 50.0  # Yeterli veri yoksa varsayılan
        
    features = ['RSI', 'ATR', 'Returns', 'Vol_Change']
    X = df_clean[features]
    y = df_clean['Target']
    
    # Train/Test Split (Backtest için veri ayrımı)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_state=42, shuffle=False)
    
    # XGBoost Model Eğitimi
    model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    # Backtest Başarı Oranı (Win Rate)
    y_pred_test = model.predict(X_test)
    win_rate = round(accuracy_score(y_test, y_pred_test) * 100, 1)
    
    # Canlı An İçin Tahmin Olasılığı (Yukarı yön olasılığı)
    son_ozellikler = X.tail(1)
    yukselis_olasiligi = float(model.predict_proba(son_ozellikler)[0][1])
    
    return yukselis_olasiligi, win_rate

def tek_hisse_analiz_et(h_kod):
    h_clean = str(h_kod).strip().upper()
    df_15m, sirket_adi = fetch_yahoo_data(h_clean, "15m", "1mo")
    df_1d, _ = fetch_yahoo_data(h_clean, "1d", "3mo")

    if df_15m is None or df_1d is None or len(df_15m) < 40 or len(df_1d) < 30: return None

    son_fiyat = float(df_15m['Close'].iloc[-1])
    sma_20 = df_15m['Close'].rolling(window=20).mean()
    std_20 = df_15m['Close'].rolling(window=20).std()
    upper_band, lower_band = (sma_20 + (std_20 * 2)).iloc[-1], (sma_20 - (std_20 * 2)).iloc[-1]
    
    vol_sum = df_15m['Volume'].rolling(14).sum()
    vwap = (df_15m['Close'] * df_15m['Volume']).rolling(14).sum() / vol_sum
    son_vwap = round(vwap.iloc[-1], 2)
    son_rsi_15m = float(compute_rsi(df_15m['Close'], 14).iloc[-1])

    close_1d = df_1d['Close']
    atr_14_1d = float(compute_atr(df_1d, 14).iloc[-1])
    atr_yuzde = (atr_14_1d / son_fiyat) * 100

    # --- XGBoost & Backtest Entegrasyonu ---
    yukselis_olasiligi, win_rate = ml_tahmin_ve_backtest(df_15m)
    
    # Olasılığa göre beklenen saatlik getiri tahmini (% cinsinden)
    beklenen_1h_getiri = round(((yukselis_olasiligi - 0.5) * 2) * (atr_14_1d / son_fiyat / 4) * 100, 2)
    hedef_1h_fiyat = round(son_fiyat * (1 + (beklenen_1h_getiri / 100)), 2)

    # Dinamik Sinyal Yapısı
    if yukselis_olasiligi >= 0.65:
        sinyal = "GÜÇLÜ AL (ML)"
        strateji_metni = f"Yapay zeka %{int(yukselis_olasiligi*100)} yükseliş ihtimali öngörüyor. (Backtest Başarısı: %{win_rate})"
    elif yukselis_olasiligi <= 0.35:
        sinyal = "SAT / DÜŞÜŞ (ML)"
        strateji_metni = f"Yapay zeka %{int((1-yukselis_olasiligi)*100)} düşüş ihtimali öngörüyor. (Backtest Başarısı: %{win_rate})"
    else:
        sinyal = "NÖTR (İZLE)"
        strateji_metni = f"Yön belirsiz. Model kararsız (%{int(yukselis_olasiligi*100)} ihtimal)."

    tahminler_yuzde = {
        '1. Gün Tahmin (%)': round(beklenen_1h_getiri * 2, 2),
        '2. Gün Tahmin (%)': round(beklenen_1h_getiri * 3.5, 2),
        '3. Gün Tahmin (%)': round(beklenen_1h_getiri * 4.5, 2)
    }

    yarin_alis = round(son_fiyat - (atr_14_1d * 0.35), 2)
    potansiyel_pik = round(upper_band + (atr_14_1d * 0.1), 2)

    return {
        'Şirket Adı': sirket_adi, 'Kod': h_clean, 'Son Fiyat (SEK)': round(son_fiyat, 2),
        '1 Saatlik Yön (%)': beklenen_1h_getiri,
        '1 Saatlik Hedef Fiyat (SEK)': hedef_1h_fiyat,
        'ML Kazanma Oranı (%)': win_rate,  # Backtest metriği
        '1. Gün Tahmin (%)': tahminler_yuzde['1. Gün Tahmin (%)'], '2. Gün Tahmin (%)': tahminler_yuzde['2. Gün Tahmin (%)'], '3. Gün Tahmin (%)': tahminler_yuzde['3. Gün Tahmin (%)'],
        'Sinyal': sinyal, 'RSI (15m)': round(son_rsi_15m, 1), 'VWAP': son_vwap,
        'Potansiyel Pik': potansiyel_pik, 'Yarınki Alış': yarin_alis, 'Strateji': strateji_metni,
        'Analiz Zamanı': datetime.now().strftime("%Y-%m-%d %H:%M")
    }

class Database:
    def __init__(self, db_name="app_data.db"):
        self.db_name = db_name
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

db = Database()
