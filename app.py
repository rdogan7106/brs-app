import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------
# SAYFA YAPILANDIRMASI & CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Avanza Pro Day-Trading Panel", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

DATA_FILE = "son_analiz.csv"

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
    .strategy-box {
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        font-size: 0.9rem;
        color: #0d47a1;
        line-height: 1.4;
    }
    @media (prefers-color-scheme: dark) {
        .stock-card {
            background-color: #1e2129;
            border-color: #2d323e;
        }
        .strategy-box {
            background-color: #1a233a;
            border-left: 5px solid #64b5f6;
            color: #e3f2fd;
        }
    }
    .badge-green { color: #2e7d32; font-weight: bold; }
    .badge-red { color: #c62828; font-weight: bold; }
    .badge-neutral { color: #f57f17; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# PRO FONKSİYONLAR: GÜN İÇİ VERİ (15 DAKİKALIK) & ANALİZ
# ---------------------------------------------------------
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Connection': 'keep-alive'
})

def tek_hisse_analiz_et(h_kod):
    h_clean = str(h_kod).strip().upper()
    h_yf = h_clean if h_clean.endswith(".ST") else f"{h_clean}.ST"
    
    # DİKKAT: 1d yerine 15m (15 dakikalık) veri çekiyoruz
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{h_yf}?range=5d&interval=15m"
    
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
        
        # Gün içi kısa periyotlu EMA'lar (Trend takibi için)
        ema_9 = close.ewm(span=9, adjust=False).mean().iloc[-1]
        ema_21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
        
        # RSI 14 (15 dakikalık periyotta)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        son_rsi = float((100 - (100 / (1 + rs))).iloc[-1])

        # BOLLINGER BANTLARI (Zirve / Pik noktasını bulmak için)
        sma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        upper_band = (sma_20 + (std_20 * 2)).iloc[-1]
        lower_band = (sma_20 - (std_20 * 2)).iloc[-1]

        # VWAP Yaklaşımı (Hacim Ağırlıklı Hareketli Ortalama)
        vwap = (df['Close'] * df['Volume']).rolling(14).sum() / df['Volume'].rolling(14).sum()
        son_vwap = vwap.iloc[-1]

        # Hacim Skoru (Son 15 dk vs Son 3 saatin ortalaması)
        vol_s = df['Volume'].dropna()
        ort_hacim = vol_s.tail(12).mean() if len(vol_s) >= 12 else 1
        hacim_kati = float(vol_s.iloc[-1] / ort_hacim) if ort_hacim > 0 else 1.0

        # ATR (15 dakikalık oynaklık)
        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_14 = float(true_range.rolling(14).mean().iloc[-1])
        
        # Dinamik Zirve ve Stop-Loss Hesaplama
        stop_loss = round(son_vwap - (atr_14 * 1.5), 2) if son_fiyat > son_vwap else round(son_fiyat - (atr_14 * 2), 2)
        potansiyel_pik = round(upper_band + (atr_14 * 0.5), 2)

        # GÜN İÇİ PİK VE SİNYAL ÜRETİMİ
        if son_fiyat >= upper_band and son_rsi > 70:
            sinyal = "PİK YAPTI (SAT)"
            strateji_metni = (
                f"🚨 **DİKKAT (ZİRVE):** Fiyat Bollinger üst bandını aştı ve RSI aşırı alımda ({round(son_rsi, 1)}). "
                f"Hisse şu an gün içi **PİK (Tepe)** noktasında olabilir. Geri çekilme riski çok yüksek. "
                f"Kârı cebe yakışır prensibiyle **{son_fiyat} SEK** civarından satış planlanmalıdır."
            )
            tahmin_yonu = -1.0 
            
        elif hacim_kati > 1.5 and son_fiyat > son_vwap and ema_9 > ema_21 and 40 <= son_rsi <= 65:
            sinyal = "GÜÇLÜ AL & TUT"
            strateji_metni = (
                f"🔥 **MOMENTUM YÜKSEK:** Hacim {round(hacim_kati, 1)}x arttı ve fiyat VWAP'ın ({round(son_vwap, 2)}) üzerinde. "
                f"Yükseliş trendi güçlü. Satış için acele etme. İlk potansiyel gün içi zirve (pik) noktası **{potansiyel_pik} SEK** seviyesidir. "
                f"🛡️ **İzleyen Stop:** {stop_loss} SEK altında saatlik kapanış gelirse pozisyonu kapat."
            )
            tahmin_yonu = 1.5
            
        elif son_fiyat < lower_band and son_rsi < 30:
            sinyal = "DİPTEN TEPKİ (AL)"
            strateji_metni = (
                f"🎯 **DİP FIRSATI:** Hisse gün içi aşırı satıldı (RSI: {round(son_rsi, 1)}). "
                f"Kısa vadeli (scalping) bir işlemle **{round(son_vwap, 2)} SEK** seviyesine kadar bir sıçrama beklenebilir. Stop: {stop_loss} SEK."
            )
            tahmin_yonu = 1.0
            
        else:
            sinyal = "NÖTR (İZLE)"
            strateji_metni = (
                f"⚖️ **YATAY SEYİR:** Fiyat ortalamalar arasında sıkışmış durumda. Net bir trend yok. "
                f"Yukarı kırılımda {round(upper_band, 2)} SEK, aşağı kırılımda {round(lower_band, 2)} SEK hedeflenebilir."
            )
            tahmin_yonu = 0.0

        # UI için Saatlik Yön Tahmini (Sembolik yüzde)
        beklenen_getiri = round((atr_14 / son_fiyat) * 100 * tahmin_yonu, 2)

        return {
            'Şirket Adı': sirket_adi,
            'Kod': h_clean,
            'Son Fiyat (SEK)': round(son_fiyat, 2),
            '1 Saatlik Yön (%)': beklenen_getiri,
            'Sinyal': sinyal,
            'RSI (14)': round(son_rsi, 1),
            'Hacim Katı': round(hacim_kati, 1),
            'VWAP': round(son_vwap, 2),
            'Potansiyel Pik': potansiyel_pik,
            'Stop-Loss': stop_loss,
            'Strateji': strateji_metni,
            'Analiz Zamanı': datetime.now().strftime("%Y-%m-%d %H:%M")
        }
    except Exception:
        return None

# ---------------------------------------------------------
# UI & STREAMLIT ARAYÜZÜ
# ---------------------------------------------------------
st.title("⚡ Avanza Pro Day-Trading Analizi")
st.caption("🚀 Gerçekçi 15-Dk ATR, VWAP Matematiği & Gün İçi Pik Zamanlamaları")

if "pro_analiz_df" not in st.session_state:
    if os.path.exists(DATA_FILE):
        st.session_state.pro_analiz_df = pd.read_csv(DATA_FILE)
    else:
        st.session_state.pro_analiz_df = None

st.sidebar.header("⚙️ Ayarlar")
gorunum_modu = st.sidebar.radio("📱 Görünüm Modu:", ["Mobil Kart Görünümü (Tavsiye)", "Klasik Masaüstü Tablosu"])

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

girdi = st.sidebar.text_area("Hisse Listeniz:", value=varsayilan_liste, height=350)
hisseler = [h.strip() for h in girdi.split(",") if h.strip()]

if st.button("🔄 Yeni Analiz Başlat / Gün İçi Verileri Çek", type="primary", use_container_width=True):
    if hisseler:
        rapor = []
        bar = st.progress(0)
        durum = st.empty()
        durum.text("⚡ Hisseler taranıyor... 15 dakikalık grafikler ve VWAP hesaplanıyor.")
        
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
            # Sıralamayı artık 1 saatlik ivmeye göre yapıyoruz
            df_res = df_res.sort_values(by='1 Saatlik Yön (%)', ascending=False)
            st.session_state.pro_analiz_df = df_res
            df_res.to_csv(DATA_FILE, index=False)

if st.session_state.pro_analiz_df is not None:
    df_res = st.session_state.pro_analiz_df
    
    son_zaman = df_res['Analiz Zamanı'].iloc[0] if 'Analiz Zamanı' in df_res.columns else "Bilinmiyor"
    st.info(f"🕒 Ekranda gördüğünüz verilerin en son güncellenme zamanı: **{son_zaman}**")
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("Başarıyla Analiz Edilen", len(df_res))
    # Metriği artık saatlik yükseliş beklentisine göre veriyoruz
    c2.metric("Saatlik Yükseliş Beklenen", len(df_res[df_res['1 Saatlik Yön (%)'] > 0]))

    if gorunum_modu == "Mobil Kart Görünümü (Tavsiye)":
        st.subheader("🔥 Canlı Aksiyon Planları & Gün İçi Tepe Noktaları")
        for _, row in df_res.iterrows():
            # Yön Rengi
            if row['1 Saatlik Yön (%)'] > 0:
                t1_color = "green"
            elif row['1 Saatlik Yön (%)'] < 0:
                t1_color = "red"
            else:
                t1_color = "neutral"

            # Sinyal Renkleri
            if "AL" in row['Sinyal']:
                bg_color, text_color = '#d4edda', '#155724'
            elif "SAT" in row['Sinyal']:
                bg_color, text_color = '#f8d7da', '#721c24'
            else:
                bg_color, text_color = '#fff3cd', '#856404'
            
            st.markdown(f"""
            <div class="stock-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin:0; font-size: 1.05rem;">{row['Şirket Adı']} <span style="font-size:0.85rem; color:#6c757d;">({row['Kod']})</span></h4>
                    <span style="background-color: {bg_color}; color: {text_color}; padding: 3px 8px; border-radius: 5px; font-weight: bold; font-size: 0.8rem;">{row['Sinyal']}</span>
                </div>
                <div style="margin-top: 8px; font-size: 0.9rem;">
                    <b>Fiyat:</b> {row['Son Fiyat (SEK)']} SEK | <b>RSI (15m):</b> {row['RSI (14)']} | <b>Hacim İvmesi:</b> {row['Hacim Katı']}x
                </div>
                <hr style="margin: 8px 0;">
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                    <span><b>1 Saatlik Yön:</b> <span class="badge-{t1_color}">%{row['1 Saatlik Yön (%)']:+.2f}</span></span>
                    <span><b>VWAP:</b> <span>{row['VWAP']} SEK</span></span>
                    <span><b>Potansiyel Pik:</b> <span class="badge-green">{row['Potansiyel Pik']} SEK</span></span>
                </div>
                <div class="strategy-box">
                    {row['Strateji']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.dataframe(df_res.drop(columns=['Strateji']), use_container_width=True)
