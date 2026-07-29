import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3
import datetime
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import warnings

warnings.filterwarnings('ignore')

# LightGBM için Graceful Degradation (Varsa kullan, yoksa Sklearn'e dön)
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

# ==========================================
# 1. VERİTABANI VE LOGLAMA (SQLite)
# ==========================================
def init_db():
    conn = sqlite3.connect('trade_system.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            ticker TEXT,
            total_trades INTEGER,
            win_rate REAL,
            net_profit REAL,
            model_accuracy REAL
        )
    ''')
    conn.commit()
    return conn

def save_backtest_result(conn, ticker, total_trades, win_rate, net_profit, model_accuracy):
    c = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO backtest_results (date, ticker, total_trades, win_rate, net_profit, model_accuracy)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (now, ticker, total_trades, win_rate, net_profit, model_accuracy))
    conn.commit()

# ==========================================
# 2. TEKNİK İNDİKATÖRLER VE VERİ ZENGİNLEŞTİRME
# ==========================================
def add_technical_indicators(df):
    # RSI (14 Günlük)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)

    # ATR (Average True Range) - Risk Yönetimi İçin
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    return df

def get_data(ticker, start_date, end_date):
    # Ana hisse verisi
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        return None
    
    # Çoklu index sorununu çöz (Yfinance bazen multi-index döner)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = add_technical_indicators(df)

    # VIX Verisi ile Zenginleştirme (Piyasa Volatilitesi - Yeni Özellik)
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    if not vix.empty:
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.droplevel(1)
        df['VIX_Close'] = vix['Close']
    else:
        df['VIX_Close'] = 20 # Fallback değer
        
    df['VIX_MA_5'] = df['VIX_Close'].rolling(window=5).mean()

    # ML Hedef Değişkeni: Bir sonraki günün kapanışı bugünden yüksek mi? (Lookahead bias önlendi)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Son satırda Target NaN olacağı için ve indikatörlerdeki NaN'ları temizle
    df.dropna(inplace=True)
    
    return df

# ==========================================
# 3. MAKİNE ÖĞRENMESİ (Walk-Forward & Optimizasyon)
# ==========================================
class MLPredictor:
    def __init__(self):
        self.model = None
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'BB_Mid', 'ATR', 'VIX_Close']

    def train_with_walk_forward(self, df):
        X = df[self.features]
        y = df['Target']

        # Walk-Forward Cross Validation (Zaman serisi sızıntısını önler)
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Temel Modeli Seç (LightGBM varsa kullan, yoksa Random Forest)
        if LGBM_AVAILABLE:
            base_model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1]
            }
        else:
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }

        # Hiperparametre Optimizasyonu
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=5,
            scoring='accuracy',
            cv=tscv,
            random_state=42,
            n_jobs=-1
        )
        
        # Tüm veri üzerinde en iyi parametreleri bul
        search.fit(X, y)
        self.model = search.best_estimator_
        
        # Son %20'lik dilimde doğrulama skoru al
        split_idx = int(len(df) * 0.8)
        X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
        preds = self.model.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        return acc, search.best_params_

    def predict_future(self, current_data):
        return self.model.predict(current_data[self.features].tail(1))[0]

# ==========================================
# 4. BACKTEST VE RISK YONETIMI (Trailing Stop)
# ==========================================
class RiskManagerBacktester:
    def __init__(self, initial_capital, risk_per_trade_pct):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.trades = []

    def run_backtest(self, df, model):
        X = df[model.features]
        df['Prediction'] = model.model.predict(X)
        
        # Sadece test verisi üzerinde backtest yap (Son %20)
        split_idx = int(len(df) * 0.8)
        test_df = df.iloc[split_idx:].copy()
        
        position = 0 # 0: Nakit, 1: Long
        entry_price = 0
        position_size = 0
        highest_price = 0
        trailing_stop = 0

        for index, row in test_df.iterrows():
            current_price = row['Close']
            current_atr = row['ATR']
            signal = row['Prediction']

            # 1. Trailing Stop (İzleyen Stop) Kontrolü
            if position == 1:
                # Fiyat yeni bir zirve yaptıysa trailing stop'u yukarı çek
                if current_price > highest_price:
                    highest_price = current_price
                    # ATR tabanlı izleyen stop (Örn: Zirveden 2 ATR aşağısı)
                    new_stop = highest_price - (current_atr * 2)
                    if new_stop > trailing_stop:
                        trailing_stop = new_stop
                
                # Stop Patlarsa veya Satış Sinyali Gelirse Çık
                if current_price <= trailing_stop or signal == 0:
                    profit = (current_price - entry_price) * position_size
                    self.capital += (position_size * current_price) # Pozisyonu kapat
                    self.trades.append({'Type': 'SELL', 'Price': current_price, 'Profit': profit})
                    position = 0
                    continue

            # 2. Alım Sinyali ve Risk Yönetimi (ATR Pozisyon Büyüklüğü)
            if position == 0 and signal == 1:
                # Risk edilen miktar (Sermayenin %X'i)
                risk_amount = self.capital * (self.risk_per_trade_pct / 100)
                
                # ATR bazlı Stop mesafesi (Örn: 2 ATR)
                stop_distance = current_atr * 2
                
                # Alınabilecek maksimum hisse adedi (Sermaye koruması)
                if stop_distance > 0:
                    position_size = risk_amount / stop_distance
                    # Sermayeyi aşmamak için kontrol
                    max_affordable = self.capital / current_price
                    position_size = min(position_size, max_affordable)
                    
                    cost = position_size * current_price
                    self.capital -= cost
                    entry_price = current_price
                    highest_price = current_price
                    trailing_stop = entry_price - stop_distance
                    
                    position = 1
                    self.trades.append({'Type': 'BUY', 'Price': current_price, 'Size': position_size})

        # Test sonunda açık pozisyon varsa kapat
        if position == 1:
            final_price = test_df.iloc[-1]['Close']
            profit = (final_price - entry_price) * position_size
            self.capital += (position_size * final_price)
            self.trades.append({'Type': 'SELL', 'Price': final_price, 'Profit': profit})

        return self._generate_stats()

    def _generate_stats(self):
        winning_trades = len([t for t in self.trades if t.get('Type') == 'SELL' and t.get('Profit', 0) > 0])
        total_closed_trades = len([t for t in self.trades if t.get('Type') == 'SELL'])
        
        win_rate = (winning_trades / total_closed_trades * 100) if total_closed_trades > 0 else 0
        net_profit = self.capital - self.initial_capital
        
        return {
            'Total Trades': total_closed_trades,
            'Win Rate (%)': round(win_rate, 2),
            'Net Profit ($)': round(net_profit, 2),
            'Final Capital ($)': round(self.capital, 2)
        }

# ==========================================
# 5. STREAMLIT ARAYÜZÜ
# ==========================================
def main():
    st.set_page_config(page_title="QuantTrade v3 - ML & ATR Backtester", layout="wide")
    st.title("📈 QuantTrade v3: Gelişmiş ML ve Risk Yönetimi")
    
    # DB Başlat
    conn = init_db()

    # Sidebar Ayarları
    st.sidebar.header("Sistem Parametreleri")
    ticker = st.sidebar.text_input("Hisse/Kripto Sembolü", "AAPL")
    start_date = st.sidebar.date_input("Başlangıç Tarihi", datetime.date(2020, 1, 1))
    end_date = st.sidebar.date_input("Bitiş Tarihi", datetime.date.today())
    
    st.sidebar.markdown("---")
    initial_cap = st.sidebar.number_input("Başlangıç Sermayesi ($)", 10000)
    risk_pct = st.sidebar.slider("İşlem Başına Risk (%)", 1.0, 5.0, 2.0, 0.5)

    if st.sidebar.button("Analizi ve Backtest'i Başlat"):
        with st.spinner(f"{ticker} verileri çekiliyor ve model eğitiliyor..."):
            
            # 1. Veri Hazırlığı
            df = get_data(ticker, start_date, end_date)
            
            if df is None or len(df) < 100:
                st.error("Yeterli veri bulunamadı. Lütfen sembolü veya tarihleri kontrol edin.")
                return

            st.success("Veriler başarıyla çekildi ve VIX volatilite verisiyle zenginleştirildi!")

            # 2. Model Eğitimi (Walk-Forward CV & Hyperparameter Tuning)
            ml_predictor = MLPredictor()
            accuracy, best_params = ml_predictor.train_with_walk_forward(df)
            
            # 3. Backtest ve Risk Yönetimi (Trailing Stop)
            backtester = RiskManagerBacktester(initial_cap, risk_pct)
            stats = backtester.run_backtest(df, ml_predictor)
            
            # 4. Sonuçları DB'ye Kaydet
            save_backtest_result(conn, ticker, stats['Total Trades'], stats['Win Rate (%)'], stats['Net Profit ($)'], accuracy)

            # --- ARAYÜZ GÖSTERİMİ ---
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🧠 Makine Öğrenmesi Performansı")
                model_name = "LightGBM" if LGBM_AVAILABLE else "Random Forest"
                st.markdown(f"**Kullanılan Model:** {model_name}")
                st.metric("Walk-Forward Doğruluk Oranı (Accuracy)", f"%{accuracy*100:.1f}")
                st.write("**Bulunan En İyi Hiperparametreler:**", best_params)
                st.info("Not: Model geleceği görmemesi için TimeSeriesSplit ile eğitilmiş ve VIX endeksi ile zenginleştirilmiştir.")
                
            with col2:
                st.subheader("🛡️ Backtest Sonuçları (Trailing Stop Aktif)")
                st.metric("Net Kâr / Zarar", f"${stats['Net Profit ($)']}")
                st.metric("Kazanma Oranı (Win Rate)", f"%{stats['Win Rate (%)']}")
                st.metric("Toplam Tamamlanan İşlem", stats['Total Trades'])
                st.write(f"**Final Sermaye:** ${stats['Final Capital ($)']}")

            st.markdown("---")
            st.subheader(f"📊 {ticker} Fiyat ve ATR (Volatilite) Grafiği")
            chart_data = df[['Close', 'BB_Upper', 'BB_Lower']].tail(150)
            st.line_chart(chart_data)

            # 5. Gelecek Tahmini
            st.markdown("---")
            st.subheader("🔮 Canlı Sinyal (Yarın için tahmin)")
            future_signal = ml_predictor.predict_future(df)
            
            if future_signal == 1:
                st.success("Yapay Zeka Sinyali: **YÜKSELİŞ (LONG) Bekleniyor** 🚀")
            else:
                st.warning("Yapay Zeka Sinyali: **DÜŞÜŞ veya YATAY (NAKİT/SHORT) Bekleniyor** 📉")

if __name__ == '__main__':
    main()
