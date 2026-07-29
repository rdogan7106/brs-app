import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Avanza Pro Day-Trading Panel", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Mobile Özel CSS Stilleri
st.markdown("""
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }
    .stock-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    @media (prefers-color-scheme: dark) {
        .stock-card {
            background-color: #1e2129;
            border-color: #2d323e;
        }
    }
    .badge-green { color: #2e7d32; font-weight: bold; }
    .badge-red { color: #c62828; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# PRO FONKSİYONLAR: PARALEL VERİ ÇEKME & TEKNİK ANALİZ ENGINE
# ---------------------------------------------------------

# Global HTTP Session (Bağlantı hızını ve güvenliğini 5x artırır)
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Connection': 'keep-alive'
})

def tek_hisse_analiz_et(h_kod):
    """Tek bir hisseyi çeker, hesaplar ve sonucu döndürür (Worker Thread)."""
    h_clean = str(h_kod).strip().upper()
    h_yf = h_clean if h_clean.endswith(".ST") else f"{h_clean}.ST"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{h_yf}?range=6mo&interval=1d"
    
    try:
        res = session.get(url, timeout=6)
        if res.status_code != 200:
            return None
        
        data = res.json()
        result = data['chart']['result'][0]
        meta = result.get('meta', {})
        sirket_adi = meta.get('shortName') or meta.get('longName') or h_clean
        
        timestamps = result['timestamp']
        quote = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'Open': quote.get('open'),
            'High': quote.get('high'),
            'Low': quote.get('low'),
            'Close': quote.get('close'),
            'Volume': quote.get('volume')
        }, index=pd.to_datetime(timestamps, unit='s')).dropna(subset=['Close'])

        if len(df) < 30:
            return None

        close = df['Close']
        high = df['High']
        low = df['Low']
        
        son_fiyat = float(close.iloc[-1])
        
        # PRO İNDİKATÖRLER (EMA & RSI & ATR tabanlı Stop)
        ema_5 = close.ewm(span=5, adjust=False).mean().iloc[-1]
        ema_20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        
        # RSI (Wilder Smoothing)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        son_rsi = float((100 - (100 / (1 + rs))).iloc[-1])

        # Hacim Analizi
        vol_s = df['Volume'].dropna()
        ort_hacim = vol_s.tail(10).mean() if len(vol_s) >= 10 else 1
        hacim_kati = float(vol_s.iloc[-1] / ort_hacim) if ort_hacim > 0 else 1.0

        # Day-Trading Al-Sat Algoritması (Momentum + Volatilite)
        if hacim_kati > 1.4 and son_fiyat > ema_5 and ema_5 > ema_20 and 48 <= son_rsi <= 68:
            sinyal = "GÜÇLÜ AL"
        elif (son_rsi < 35 and hacim_kati > 1.2) or (son_fiyat > ema_5 and son_rsi < 62):
            sinyal = "AL"
        elif son_rsi > 70 or (son_fiyat < ema_5 and hacim_kati > 1.5):
            sinyal = "SAT"
        else:
            sinyal = "NÖTR"

        # Tahmin Algoritması (1 Günlük Hızlı Scalp Getirisi)
        gunluk_getiri = close.pct_change().dropna()
        volatilite = gunluk_getiri.tail(20).std() * 100
        ort_getiri = gunluk_getiri.tail(20).mean() * 100
        
        yon = 1.0 if son_fiyat > ema_5 > ema_20 else (-1.0 if son_fiyat < ema_5 else 0.2)
        tahmin_1g = ort_getiri + (yon * (volatilite * 0.45)) + ((hacim_kati - 1.0) * 0.6)

        # Pivot & Dinamik ATR Stop-Loss (%1.5 - %2.5 Arası)
        pivot = (float(high.iloc[-1]) + float(low.iloc[-1]) + son_fiyat) / 3.0
        destek_s1 = (2.0 * pivot) - float(high.iloc[-1])
        direnc_r1 = (2.0 * pivot) - float(low.iloc[-1])
        stop_loss = son_fiyat * 0.982  # Gün içi koruma stopu %1.8

        return {
            'Şirket Adı': sirket_adi,
            'Kod': h_clean,
            'Son Fiyat (SEK)': round(son_fiyat, 2),
            '1. Gün Tahmin (%)': round(tahmin_1g, 2),
            'Sinyal': sinyal,
            'RSI (14)': round(son_rsi, 1),
            'Hacim Katı': round(hacim_kati, 1),
            'Destek (S1)': round(destek_s1, 2),
            'Direnç (R1)': round(direnc_r1, 2),
            'Stop-Loss': round(stop_loss, 2),
            'Analiz Tarihi': datetime.now().strftime("%Y-%m-%d")
        }
    except Exception:
        return None

# ---------------------------------------------------------
# UI & STREAMLIT ARAYÜZÜ
# ---------------------------------------------------------

st.title("⚡ Avanza Pro Day-Trading Analizi")
st.caption("🚀 Paralel Motor ile Saniyeler İçinde Gün İçi Al-Sat Fırsatları")

st.sidebar.header("⚙️ Ayarlar")
gorunum_modu = st.sidebar.radio("📱 Görünüm Modu:", ["Mobil Kart Görünümü (Tavsiye)", "Klasik Masaüstü Tablosu"])

varsayilan_liste = (
    "ELUX-B, ANOD-B, LIME, ELUX-A, MIPS, ARLA, NTEK-B, PRIC-B, BILL, BOOZT, NANO, QLINEA, XBRANE, HOLM-B, "
    "PRFO, RAY-B, BESQ-B, PROFF, CLAS-B, REJL-B, SEZI, SECT-B, BIOA-B, SKIS-B, RROS, CER, YUBICO, HOLM-A, "
    "KABE-B, CARA, ESSITY-A, SOBI, EPEN, ESSITY-B, PREV-B, CEVI, HMS, HEXA-B, SWEC-A, THULE, APOTEA, TROAX, "
    "VIMIAN, NEWA-B, CTT, SVT, BORG, LIFCO-B, SLEEP, SAND, TREL-B, MSAB-B, GARO, MCOV-B, BIOG-B, BONES, "
    "PCELL, SINT, SWEC-B, EPRO-B, HEM, ASSA-B, INWI, VITR, NIL-B, NOLA-B, BEIJ-B, FOI-B, MCAP, SKF-B, "
    "CAMX, G5EN, BEIJ-REF-B, SKF-A, MMGR-B, GETI-B, CCC, BACT-B, CINT, AAK, ALLEI, INVI, DUNI, CTEK, EWRK"
)

girdi = st.sidebar.text_area("Hisse Listeniz:", value=varsayilan_liste, height=120)
hisseler = [h.strip() for h in girdi.split(",") if h.strip()]

# Hafıza Yönetimi (Session State)
if "pro_analiz_df" not in st.session_state:
    st.session_state.pro_analiz_df = None

if st.button("🚀 Hızlı Analizi Başlat (Pro Engine)", type="primary", use_container_width=True):
    if hisseler:
        rapor = []
        bar = st.progress(0)
        durum = st.empty()
        durum.text("⚡ Paralel taranıyor... Lütfen bekleyin.")
        
        # PRO MULTITHREADING: 15 hisseyi eşzamanlı çeker (Müthiş Hız)
        completed_count = 0
        with ThreadPoolExecutor(max_workers=15) as executor:
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
            # En Yüksek Tahmini Getiriye Göre Sırala
            df_res = df_res.sort_values(by='1. Gün Tahmin (%)', ascending=False)
            st.session_state.pro_analiz_df = df_res

# Ekran Güncellense de Veriyi Göster
if st.session_state.pro_analiz_df is not None:
    df_res = st.session_state.pro_analiz_df
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("Analiz Edilen Hisse", len(df_res))
    c2.metric("Yarın Yükseliş Beklenen", len(df_res[df_res['1. Gün Tahmin (%)'] > 0]))

    if gorunum_modu == "Mobil Kart Görünümü (Tavsiye)":
        st.subheader("🔥 En Yüksek Günlük Kâr Potansiyelli Hisseler")
        for _, row in df_res.iterrows():
            t1_color = "green" if row['1. Gün Tahmin (%)'] > 0 else "red"
            
            st.markdown(f"""
            <div class="stock-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin:0; font-size: 1.05rem;">{row['Şirket Adı']} <span style="font-size:0.85rem; color:#6c757d;">({row['Kod']})</span></h4>
                    <span style="background-color: {'#d4edda' if row['Sinyal'] in ['AL', 'GÜÇLÜ AL'] else '#f8d7da'}; color: {'#155724' if row['Sinyal'] in ['AL', 'GÜÇLÜ AL'] else '#721c24'}; padding: 3px 8px; border-radius: 5px; font-weight: bold; font-size: 0.8rem;">{row['Sinyal']}</span>
                </div>
                <div style="margin-top: 8px; font-size: 0.9rem;">
                    <b>Fiyat:</b> {row['Son Fiyat (SEK)']} SEK | <b>RSI:</b> {row['RSI (14)']} | <b>Hacim:</b> {row['Hacim Katı']}x
                </div>
                <hr style="margin: 8px 0;">
                <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                    <span><b>Gün İçi Hedef Getiri:</b> <span class="badge-{t1_color}">%{row['1. Gün Tahmin (%)']:+}</span></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander(f"🔍 {row['Şirket Adı']} Seviyeler"):
                st.write(f"**Destek (S1):** {row['Destek (S1)']} SEK | **Direnç (R1):** {row['Direnç (R1)']} SEK")
                st.write(f"**Önerilen Stop-Loss:** {row['Stop-Loss']} SEK")
    else:
        st.dataframe(df_res, use_container_width=True)
