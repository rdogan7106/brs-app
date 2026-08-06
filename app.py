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

st.set_page_config(page_title="Borsa Panel", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
set_page_style()

st.sidebar.markdown("## ⚙️ Kontrol Paneli")
girdi = st.sidebar.text_area("📋 Hisse Listeniz:", value=VARSAYILAN_LISTE, height=300)
hisseler = [h.strip() for h in girdi.split(",") if h.strip()]

@st.cache_data(ttl=3600, show_spinner=False)
def get_ticker_name_map(ticker_tuple):
    name_map = {}
    def _fetch(t):
        h_yf = f"{t.upper()}.ST" if not t.upper().endswith(".ST") else t.upper()
        try:
            res = session.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{h_yf}?range=1d&interval=1d", timeout=5)
            if res.status_code == 200:
                name = res.json()['chart']['result'][0].get('meta', {}).get('shortName') or t
                return t, name
        except: pass
        return t, t.upper()
    with ThreadPoolExecutor(max_workers=20) as executor:
        for future in as_completed([executor.submit(_fetch, t) for t in list(ticker_tuple)]):
            t, name = future.result()
            name_map[t] = name
    return name_map

ticker_names = get_ticker_name_map(tuple(hisseler)) if hisseler else {}
def format_hisse(ticker): return ticker_names.get(ticker, ticker)

st.sidebar.caption(f"📊 Listedeki hisse sayısı: **{len(hisseler)}**")
ml_backend = "LightGBM" if HAS_LIGHTGBM else ("XGBoost" if HAS_XGBOOST else ("sklearn-GradientBoosting" if HAS_SKLEARN else "Yok"))
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Sistem Durumu")
st.sidebar.markdown(f"- **ML Backend:** `{ml_backend}`")
st.sidebar.markdown("- **Database:** `SQLite` ✓\n- **Veri Kaynağı:** `Yahoo Finance` ✓")
if ml_backend == "Yok": st.sidebar.warning("ML için: `pip install lightgbm xgboost scikit-learn`")

st.markdown("# ⚡ Borsa Panel")
st.markdown("*15m Zirve Avcısı · ML Tahmin · Backtest · Risk Yönetimi · Sharpe Metrikleri*")
st.markdown("---")

tab_analiz, tab_backtest, tab_ml, tab_risk, tab_gecmis = st.tabs(["📊 Canlı Analiz", "🔬 Backtest", "🤖 ML Tahmin", "🛡️ Risk Yönetimi", "📜 Geçmiş"])

with tab_analiz:
    gorunum_modu = st.radio("Görünüm Modu:", ["Kart Görünümü", "Tablo Görünümü"], horizontal=True, label_visibility="collapsed")
    if "pro_analiz_df" not in st.session_state:
        st.session_state.pro_analiz_df = pd.read_csv(DATA_FILE) if os.path.exists(DATA_FILE) else None

    if st.button("🔄 Hibrid Analizi Başlat (Day + Swing Trade)", type="primary", use_container_width=True):
        if hisseler:
            rapor = []
            bar, durum = st.progress(0), st.empty()
            durum.text("⚡ Hisseler taranıyor... Çift Katmanlı Veri İşleniyor...")
            
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = {executor.submit(tek_hisse_analiz_et, h): h for h in hisseler}
                for i, future in enumerate(as_completed(futures)):
                    res = future.result()
                    if res: rapor.append(res)
                    bar.progress((i + 1) / len(hisseler))
            
            durum.empty(); bar.empty()
            if rapor:
                df_res = pd.DataFrame(rapor).sort_values(by=['1. Gün Tahmin (%)', '1 Saatlik Yön (%)'], ascending=[False, False])
                st.session_state.pro_analiz_df = df_res
                df_res.to_csv(DATA_FILE, index=False)
                for r in rapor: 
                    try: db.save_prediction(r)
                    except: pass

    if st.session_state.get('pro_analiz_df') is not None:
        df_res = st.session_state.pro_analiz_df
        st.info(f"🕒 Son güncelleme: **{df_res['Analiz Zamanı'].iloc[0] if 'Analiz Zamanı' in df_res.columns else 'Bilinmiyor'}**")
        st.markdown("---")
        if gorunum_modu == "Kart Görünümü":
            for _, row in df_res.iterrows():
                with st.container():
                    badge_cls = "signal-buy" if "AL" in row['Sinyal'] else ("signal-sell" if "SAT" in row['Sinyal'] else "signal-hold")
                    badge_emoji = "✅" if "AL" in row['Sinyal'] else ("❌" if "SAT" in row['Sinyal'] else "⏸️")
                    st.markdown(f"<span class='signal-badge {badge_cls}'>{badge_emoji} {row['Sinyal']}</span>", unsafe_allow_html=True)
                    st.markdown(f"**{row['Şirket Adı']}** <span style='font-size:0.75rem; color:gray;'>({row['Kod']})</span> &nbsp;<span style='font-size:0.8rem;'>Fiyat: {row['Son Fiyat (SEK)']} SEK | RSI: {row['RSI (15m)']} | VWAP: {row['VWAP']} SEK | Pik: {row['Potansiyel Pik']} SEK</span>", unsafe_allow_html=True)
                    
                    m0, m1, m2, m3, m4 = st.columns(5)
                    # YENI OZELLIK: 1 SAATLIK HEDEF FIYAT GORUNUMU
                    m0.metric("⏱️ 1 Saat Hedefi", f"{row.get('1 Saatlik Hedef Fiyat (SEK)', row['Son Fiyat (SEK)'])} SEK", f"%{row['1 Saatlik Yön (%)']:+.2f}")
                    m1.metric("1 Saat Yön", f"%{row['1 Saatlik Yön (%)']:+.2f}")
                    m2.metric("1. Gün", f"%{row['1. Gün Tahmin (%)']:+.2f}")
                    m3.metric("2. Gün", f"%{row['2. Gün Tahmin (%)']:+.2f}")
                    m4.metric("3. Gün", f"%{row['3. Gün Tahmin (%)']:+.2f}")
                    
                    st.markdown(f"🎯 **Strateji:** {row['Strateji']}\n📅 **Alış Tavsiyesi:** {row['Yarınki Alış']} SEK seviyesi takip edilebilir.")
                    st.markdown("---")
        else:
            st.dataframe(df_res.drop(columns=['Strateji']), use_container_width=True)

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
        if df_bt is None or len(df_bt) < 100: st.error("Yetersiz veri.")
        else:
            ml_pred = None
            if bt_strategy == "ml":
                ml_pred = MLPredictor()
                train_res = ml_pred.train(df_bt)
                if "error" in train_res:
                    st.warning("ML hatası, kural tabanlıya geçildi."); bt_strategy = "rule"; ml_pred = None
                else: st.success(f"Eğitildi: {ml_pred.model_name} | Acc: %{train_res['accuracy']*100:.1f}")
            
            result = Backtester(RiskManager(risk_per_trade=bt_risk/100)).run(df_bt, bt_hisse, bt_capital, bt_strategy, ml_pred)
            if "error" in result: st.error(result["error"])
            else:
                metrics = result['metrics']
                run_id = db.save_backtest_run(metrics)
                for t in result['trades']: db.save_backtest_trade(run_id, t)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("💰 Toplam Getiri", f"%{metrics['total_return']:+.2f}")
                m2.metric("📊 Sharpe Ratio", f"{metrics['sharpe']:.3f}")
                m3.metric("📉 Max Drawdown", f"%{metrics['max_drawdown']:.2f}")
                m4.metric("🎯 Win Rate", f"%{metrics['win_rate']:.1f}")
                
                st.line_chart(pd.DataFrame({'Tarih': result['equity_dates'], 'Portfolio': metrics['equity_curve']}).set_index('Tarih'), use_container_width=True)
                if result['trades']: st.dataframe(pd.DataFrame(result['trades'])[['entry_date', 'entry_price', 'exit_date', 'exit_price', 'shares', 'pnl', 'return_pct', 'reason']], use_container_width=True)

with tab_ml:
    st.markdown("### 🤖 ML Yön Tahmin Modeli")
    if ml_backend == "Yok": st.error("ML kütüphanesi kurulu değil.")
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
                    st.success(f"✅ {res['model']} | Accuracy: %{res['accuracy']*100:.1f}")
                    pred = predictor.predict_latest(df_ml)
                    if pred:
                        st.info(f"**{ml_hisse}** Son Tahmin: {'📈 YÜKSELİŞ' if pred['direction']==1 else '📉 DÜŞÜŞ'} | Güven: %{pred['probability']*100:.1f}")
                        db.save_ml_prediction(ml_hisse, pred['direction'], pred['probability'], res['model'], res['accuracy'], res.get('feature_importance', {}))
                    if res.get('feature_importance'):
                        fi_df = pd.DataFrame([{'Feature': k, 'Importance': v} for k, v in res['feature_importance'].items()]).sort_values(by="Importance", ascending=False)
                        st.bar_chart(fi_df.set_index('Feature'), use_container_width=True)
                else: st.error(res["error"])

with tab_risk:
    st.markdown("### 🛡️ Risk Yönetimi Hesaplayıcı")
    # Risk sekmesi orijinal kodu (Kısaltılarak korundu)
    rk_hisse = st.selectbox("Hisse Seç:", options=hisseler or ["ELUX-B"])
    if st.button("📥 Veri Çek", key="rk_fetch"):
        df_rk, _ = fetch_yahoo_data(rk_hisse, "1d", "3mo")
        if df_rk is not None:
            st.session_state.rk_price = round(float(df_rk['Close'].iloc[-1]), 2)
            from indicators import compute_atr; st.session_state.rk_atr = round(float(compute_atr(df_rk, 14).iloc[-1]), 2)
            st.success(f"Fiyat: {st.session_state.rk_price} SEK | ATR: {st.session_state.rk_atr}")
    
    p1, p2 = st.columns(2)
    rk_port = p1.number_input("Portföy Değeri:", value=100000)
    rk_price = p1.number_input("Giriş Fiyatı:", value=st.session_state.get('rk_price', 100.0))
    rk_risk = p2.slider("Risk (%)", 1, 5, 2)
    rk_atr = p2.number_input("ATR:", value=st.session_state.get('rk_atr', 3.0))
    
    if st.button("Hesapla", type="primary"):
        shares, sl, tp = RiskManager(risk_per_trade=rk_risk/100).calculate_position_size(rk_port, rk_price, rk_atr)
        st.info(f"Hisse: {shares} | Stop: {sl} | Hedef: {tp}")

with tab_gecmis:
    st.markdown("### 📜 Geçmiş Kayıtlar (SQLite)")
    sec = st.radio("Tip:", ["Tahminler", "Backtest Sonuçları", "ML Tahminleri"], horizontal=True)
    if sec == "Tahminler": st.dataframe(db.get_prediction_history(limit=100))
    elif sec == "Backtest Sonuçları": st.dataframe(db.get_backtest_history(limit=50))
    else: st.dataframe(db.get_ml_history(limit=50))
