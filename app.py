import streamlit as st
import pandas as pd
import warnings
import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import set_page_style, DATA_FILE, VARSAYILAN_LISTE
from database import db
from ml_model import MLPredictor, HAS_LIGHTGBM, HAS_XGBOOST, HAS_SKLEARN
from risk_manager import RiskManager
from backtester import Backtester
from data_fetcher import fetch_yahoo_data, tek_hisse_analiz_et, session

warnings.filterwarnings("ignore")

# Streamlit sayfa konfigürasyonu (İlki Streamlit komutu olmalıdır)
st.set_page_config(
    page_title="Borsa Panel", 
    page_icon="⚡", 
    layout="wide", 
    initial_sidebar_state="expanded"
)
set_page_style()

# --- SIDEBAR (KONTROL PANELİ) ---
st.sidebar.markdown("## ⚙️ Kontrol Paneli")
girdi = st.sidebar.text_area("📋 Hisse Listeniz (Virgülle ayırın):", value=VARSAYILAN_LISTE, height=250)
hisseler = [h.strip().upper() for h in girdi.split(",") if h.strip()]

@st.cache_data(ttl=3600, show_spinner=False)
def get_ticker_name_map(ticker_tuple):
    name_map = {}
    def _fetch(t):
        h_yf = f"{t}.ST" if not t.endswith(".ST") else t
        try:
            res = session.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{h_yf}?range=1d&interval=1d", timeout=5)
            if res.status_code == 200:
                name = res.json()['chart']['result'][0].get('meta', {}).get('shortName') or t
                return t, name
        except Exception:
            pass
        return t, t
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(_fetch, t) for t in list(ticker_tuple)]
        for future in as_completed(futures):
            t, name = future.result()
            name_map[t] = name
    return name_map

ticker_names = get_ticker_name_map(tuple(hisseler)) if hisseler else {}
def format_hisse(ticker): 
    return ticker_names.get(ticker, ticker)

st.sidebar.caption(f"📊 Listedeki hisse sayısı: **{len(hisseler)}**")

ml_backend = "LightGBM" if HAS_LIGHTGBM else ("XGBoost" if HAS_XGBOOST else ("sklearn-GradientBoosting" if HAS_SKLEARN else "Yok"))
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Sistem Durumu")
st.sidebar.markdown(f"- **ML Backend:** `{ml_backend}`")
st.sidebar.markdown("- **Database:** `SQLite` ✓\n- **Veri Kaynağı:** `Yahoo Finance` ✓")
if ml_backend == "Yok": 
    st.sidebar.warning("ML için: `pip install lightgbm xgboost scikit-learn`")

# --- ANA EKRAN HEADER ---
st.markdown("# ⚡ Borsa Panel")
st.markdown("*15m Zirve Avcısı · ML Tahmin · Backtest · Risk Yönetimi · Sharpe Metrikleri*")
st.markdown("---")

tab_analiz, tab_backtest, tab_ml, tab_risk, tab_gecmis = st.tabs([
    "📊 Canlı Analiz", "🔬 Backtest", "🤖 ML Tahmin", "🛡️ Risk Yönetimi", "📜 Geçmiş"
])

# --- TAB 1: CANLI ANALİZ ---
with tab_analiz:
    gorunum_modu = st.radio("Görünüm Modu:", ["Kart Görünümü", "Tablo Görünümü"], horizontal=True, label_visibility="collapsed")
    
    if "pro_analiz_df" not in st.session_state:
        st.session_state.pro_analiz_df = pd.read_csv(DATA_FILE) if os.path.exists(DATA_FILE) else None

    if st.button("🔄 Hibrid Analizi Başlat (Day + Swing Trade)", type="primary", use_container_width=True):
        if hisseler:
            rapor = []
            bar = st.progress(0)
            durum = st.empty()
            durum.text("⚡ Hisseler taranıyor... Çift Katmanlı Veri İşleniyor...")
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(tek_hisse_analiz_et, h): h for h in hisseler}
                for i, future in enumerate(as_completed(futures)):
                    try:
                        res = future.result()
                        if res and isinstance(res, dict): 
                            rapor.append(res)
                    except Exception as e:
                        pass
                    bar.progress((i + 1) / len(hisseler))
            
            durum.empty()
            bar.empty()
            
            if rapor:
                df_res = pd.DataFrame(rapor)
                # Güvenli Sıralama (Kolon varlığı kontrolü)
                sort_cols = [c for c in ['1. Gün Tahmin (%)', '1 Saatlik Yön (%)'] if c in df_res.columns]
                if sort_cols:
                    df_res = df_res.sort_values(by=sort_cols, ascending=False)
                
                st.session_state.pro_analiz_df = df_res
                df_res.to_csv(DATA_FILE, index=False)
                
                for r in rapor: 
                    try: 
                        db.save_prediction(r)
                    except Exception: 
                        pass
            else:
                st.warning("Hiçbir hisse için analiz verisi alınamadı.")

    if st.session_state.get('pro_analiz_df') is not None and not st.session_state.pro_analiz_df.empty:
        df_res = st.session_state.pro_analiz_df
        son_guncelleme = df_res['Analiz Zamanı'].iloc[0] if 'Analiz Zamanı' in df_res.columns else 'Bilinmiyor'
        st.info(f"🕒 Son güncelleme: **{son_guncelleme}**")
        st.markdown("---")
        
        if gorunum_modu == "Kart Görünümü":
            for _, row in df_res.iterrows():
                with st.container():
                    sinyal = str(row.get('Sinyal', 'NÖTR'))
                    badge_cls = "signal-buy" if "AL" in sinyal else ("signal-sell" if "SAT" in sinyal else "signal-hold")
                    badge_emoji = "✅" if "AL" in sinyal else ("❌" if "SAT" in sinyal else "⏸️")
                    
                    st.markdown(f"<span class='signal-badge {badge_cls}'>{badge_emoji} {sinyal}</span>", unsafe_allow_html=True)
                    st.markdown(
                        f"**{row.get('Şirket Adı', row.get('Kod', 'N/A'))}** "
                        f"<span style='font-size:0.75rem; color:gray;'>({row.get('Kod', '')})</span> &nbsp;"
                        f"<span style='font-size:0.8rem;'>Fiyat: {row.get('Son Fiyat (SEK)', 0)} SEK | "
                        f"RSI: {row.get('RSI (15m)', 0)} | VWAP: {row.get('VWAP', 0)} SEK | "
                        f"Pik: {row.get('Potansiyel Pik', 0)} SEK</span>", 
                        unsafe_allow_html=True
                    )
                    
                    m0, m1, m2, m3, m4 = st.columns(5)
                    m0.metric("⏱️ 1 Saat Hedefi", f"{row.get('1 Saatlik Hedef Fiyat (SEK)', row.get('Son Fiyat (SEK)', 0))} SEK", f"%{row.get('1 Saatlik Yön (%)', 0):+.2f}")
                    m1.metric("1 Saat Yön", f"%{row.get('1 Saatlik Yön (%)', 0):+.2f}")
                    m2.metric("1. Gün", f"%{row.get('1. Gün Tahmin (%)', 0):+.2f}")
                    m3.metric("2. Gün", f"%{row.get('2. Gün Tahmin (%)', 0):+.2f}")
                    m4.metric("3. Gün", f"%{row.get('3. Gün Tahmin (%)', 0):+.2f}")
                    
                    st.markdown(f"🎯 **Strateji:** {row.get('Strateji', 'Belirtilmedi')}\n📅 **Alış Tavsiyesi:** {row.get('Yarınki Alış', 0)} SEK seviyesi takip edilebilir.")
                    st.markdown("---")
        else:
            drop_cols = [c for c in ['Strateji'] if c in df_res.columns]
            st.dataframe(df_res.drop(columns=drop_cols), use_container_width=True)

# --- TAB 2: BACKTEST ---
with tab_backtest:
    st.markdown("### 🔬 Backtest Sistemi")
    c1, c2, c3 = st.columns(3)
    bt_hisse = c1.selectbox("Hisse Seç:", options=hisseler or ["ELUX-B"], format_func=format_hisse)
    bt_capital = c2.number_input("Başlangıç Sermayesi (SEK):", value=10000, min_value=1000, step=1000)
    bt_strategy = c3.selectbox("Strateji:", ["rule", "ml"], format_func=lambda x: "📊 Kural Bazlı" if x == "rule" else "🤖 ML Tahminli")
    
    c4, c5 = st.columns(2)
    bt_range = c4.selectbox("Veri Aralığı:", ["6mo", "1y", "2y", "5y"], index=1)
    bt_risk = c5.slider("Risk / İşlem (%):", 1, 5, 2)
    
    if st.button("🚀 Backtest Çalıştır", type="primary", use_container_width=True):
        df_bt, _ = fetch_yahoo_data(bt_hisse, "1d", bt_range, True)
        if df_bt is None or len(df_bt) < 100: 
            st.error("Yetersiz veri. Farklı bir zaman aralığı veya hisse seçin.")
        else:
            ml_pred = None
            if bt_strategy == "ml":
                ml_pred = MLPredictor()
                train_res = ml_pred.train(df_bt)
                if "error" in train_res:
                    st.warning("ML model eğitimi başarısız oldu, kural tabanlı stratejiye geçiliyor.")
                    bt_strategy = "rule"
                    ml_pred = None
                else: 
                    st.success(f"Model Eğitildi: {ml_pred.model_name} | Acc: %{train_res.get('accuracy', 0)*100:.1f}")
            
            result = Backtester(RiskManager(risk_per_trade=bt_risk/100)).run(df_bt, bt_hisse, bt_capital, bt_strategy, ml_pred)
            if "error" in result: 
                st.error(result["error"])
            else:
                metrics = result['metrics']
                try:
                    run_id = db.save_backtest_run(metrics)
                    for t in result['trades']: 
                        db.save_backtest_trade(run_id, t)
                except Exception:
                    pass
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("💰 Toplam Getiri", f"%{metrics.get('total_return', 0):+.2f}")
                m2.metric("📊 Sharpe Ratio", f"{metrics.get('sharpe', 0):.3f}")
                m3.metric("📉 Max Drawdown", f"%{metrics.get('max_drawdown', 0):.2f}")
                m4.metric("🎯 Win Rate", f"%{metrics.get('win_rate', 0):.1f}")
                
                if 'equity_dates' in result and 'equity_curve' in metrics:
                    st.line_chart(pd.DataFrame({'Tarih': result['equity_dates'], 'Portfolio': metrics['equity_curve']}).set_index('Tarih'), use_container_width=True)
                
                if result.get('trades'): 
                    st.dataframe(pd.DataFrame(result['trades'])[['entry_date', 'entry_price', 'exit_date', 'exit_price', 'shares', 'pnl', 'return_pct', 'reason']], use_container_width=True)

# --- TAB 3: ML TAHMİN ---
with tab_ml:
    st.markdown("### 🤖 ML Yön Tahmin Modeli")
    if ml_backend == "Yok": 
        st.error("ML kütüphanesi kurulu değil. `scikit-learn`, `lightgbm` veya `xgboost` paketlerinden en az birini kurun.")
    else:
        c1, c2 = st.columns(2)
        ml_hisse = c1.selectbox("Hisse (ML):", options=hisseler or ["ELUX-B"], format_func=format_hisse)
        ml_range = c2.selectbox("Aralık:", ["6mo", "1y", "2y", "5y"], index=2)
        
        if st.button("🎯 Model Eğit & Tahmin Et", type="primary", use_container_width=True):
            df_ml, _ = fetch_yahoo_data(ml_hisse, "1d", ml_range, True)
            if df_ml is not None:
                predictor = MLPredictor()
                res = predictor.train(df_ml)
                if "error" not in res:
                    st.success(f"✅ {res.get('model', 'Model')} | Accuracy: %{res.get('accuracy', 0)*100:.1f}")
                    pred = predictor.predict_latest(df_ml)
                    if pred:
                        yon_text = '📈 YÜKSELİŞ' if pred.get('direction') == 1 else '📉 DÜŞÜŞ'
                        st.info(f"**{ml_hisse}** Son Tahmin: {yon_text} | Güven: %{pred.get('probability', 0)*100:.1f}")
                        try:
                            db.save_ml_prediction(ml_hisse, pred['direction'], pred['probability'], res['model'], res['accuracy'], res.get('feature_importance', {}))
                        except Exception:
                            pass
                    
                    if res.get('feature_importance'):
                        fi_df = pd.DataFrame([{'Feature': k, 'Importance': v} for k, v in res['feature_importance'].items()]).sort_values(by="Importance", ascending=False)
                        st.bar_chart(fi_df.set_index('Feature'), use_container_width=True)
                else: 
                    st.error(res["error"])

# --- TAB 4: RİSK YÖNETİMİ ---
with tab_risk:
    st.markdown("### 🛡️ Risk Yönetimi Hesaplayıcı")
    rk_hisse = st.selectbox("Hisse Seç:", options=hisseler or ["ELUX-B"], format_func=format_hisse)
    
    if st.button("📥 Veri Çek", key="rk_fetch"):
        df_rk, _ = fetch_yahoo_data(rk_hisse, "1d", "3mo")
        if df_rk is not None and not df_rk.empty:
            st.session_state.rk_price = round(float(df_rk['Close'].iloc[-1]), 2)
            from indicators import compute_atr
            atr_series = compute_atr(df_rk, 14)
            st.session_state.rk_atr = round(float(atr_series.iloc[-1]), 2)
            st.success(f"Fiyat: {st.session_state.rk_price} SEK | ATR: {st.session_state.rk_atr}")
        else:
            st.error("Veri alınamadı.")
    
    p1, p2 = st.columns(2)
    rk_port = p1.number_input("Portföy Değeri (SEK):", value=100000)
    rk_price = p1.number_input("Giriş Fiyatı (SEK):", value=st.session_state.get('rk_price', 100.0))
    rk_risk = p2.slider("Risk (%)", 1, 5, 2)
    rk_atr = p2.number_input("ATR:", value=st.session_state.get('rk_atr', 3.0))
    
    if st.button("Hesapla", type="primary"):
        shares, sl, tp = RiskManager(risk_per_trade=rk_risk/100).calculate_position_size(rk_port, rk_price, rk_atr)
        st.info(f"Alınabilir Adet: **{shares}** | Stop-Loss: **{sl} SEK** | Hedef Fiyat: **{tp} SEK**")

# --- TAB 5: GEÇMİŞ ---
with tab_gecmis:
    st.markdown("### 📜 Geçmiş Kayıtlar (SQLite)")
    sec = st.radio("Tip:", ["Tahminler", "Backtest Sonuçları", "ML Tahminleri"], horizontal=True)
    try:
        if sec == "Tahminler": 
            st.dataframe(db.get_prediction_history(limit=100), use_container_width=True)
        elif sec == "Backtest Sonuçları": 
            st.dataframe(db.get_backtest_history(limit=50), use_container_width=True)
        else: 
            st.dataframe(db.get_ml_history(limit=50), use_container_width=True)
    except Exception as e:
        st.warning(f"Geçmiş veriler okunamadı: {e}")
