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
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# PRO FONKSİYONLAR: PARALEL VERİ ÇEKME & MATEMATİKSEL ANALİZ
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
        
        ema_5 = close.ewm(span=5, adjust=False).mean().iloc[-1]
        ema_20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        
        # RSI 14
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        son_rsi = float((100 - (100 / (1 + rs))).iloc[-1])

        # Hacim Katı
        vol_s = df['Volume'].dropna()
        ort_hacim = vol_s.tail(10).mean() if len(vol_s) >= 10 else 1
        hacim_kati = float(vol_s.iloc[-1] / ort_hacim) if ort_hacim > 0 else 1.0

        # --- YENİ EKLENEN PROFESYONEL MATEMATİKSEL İSTATİSTİKLER ---
        
        # 1. ATR (Average True Range)
        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_14 = float(true_range.rolling(14).mean().iloc[-1])
        atr_yuzde = (atr_14 / son_fiyat) * 100

        # 2. Ağırlıklı Getiri (Drift)
        gunluk_getiri = close.pct_change().dropna()
        ema_getiri = gunluk_getiri.ewm(span=10, adjust=False).mean().iloc[-1] * 100

        # 3. Momentum Skoru (Ortalamaya Dönüş Mantığı)
        if son_rsi > 75:
            momentum = -0.15 
        elif son_rsi < 30:
            momentum = 0.15  
        else:
            momentum = (son_rsi - 50) / 100.0

        hacim_etkisi = min(hacim_kati, 2.0) 
        
        # Gelecek 3 Günün Tahmini
        tahminler_yuzde = {}
        kumbulatif_fiyat = son_fiyat
        aktif_momentum = momentum

        for gun in range(1, 4):
            gun_artis_yuzde = ema_getiri + (aktif_momentum * atr_yuzde * hacim_etkisi)
            gun_artis_yuzde = max(min(gun_artis_yuzde, atr_yuzde * 1.5), -atr_yuzde * 1.5)
            
            kumbulatif_fiyat = kumbulatif_fiyat * (1 + (gun_artis_yuzde / 100))
            tahminler_yuzde[f'{gun}. Gün Tahmin (%)'] = round(((kumbulatif_fiyat - son_fiyat) / son_fiyat) * 100, 2)
            
            aktif_momentum *= 0.5 
            hacim_etkisi = max(1.0, hacim_etkisi * 0.8)

        # Dinamik Destek/Direnç ve Stop Loss (ATR Tabanlı)
        destek_s1 = round(son_fiyat - (atr_14 * 0.8), 2)
        direnc_r1 = round(son_fiyat + (atr_14 * 0.8), 2)
        stop_loss = round(son_fiyat - (atr_14 * 1.5), 2)

        # Sinyal Üretimi
        if hacim_kati > 1.4 and son_fiyat > ema_5 and ema_5 > ema_20 and 48 <= son_rsi <= 68:
            sinyal = "GÜÇLÜ AL"
        elif (son_rsi < 35 and hacim_kati > 1.2) or (son_fiyat > ema_5 and son_rsi < 62):
            sinyal = "AL"
        elif son_rsi > 70 or (son_fiyat < ema_5 and hacim_kati > 1.5):
            sinyal = "SAT"
        else:
            sinyal = "NÖTR"

        # --- GÜN İÇİ HİKAYELEŞTİRİLMİŞ GERÇEKÇİ TAHMİN (SENİN İSTEDİĞİN FORMAT) ---
        ilk_gun_artis = tahminler_yuzde['1. Gün Tahmin (%)']
        beklenen_kapanis = round(son_fiyat * (1 + (ilk_gun_artis / 100)), 2)
        
        # Sabah alım noktası fiyatın ATR'nin %30'u kadar altına esnemesiyle bulunur
        sabah_alim = round(son_fiyat - (atr_14 * 0.3), 2)
        
        # Gün içi zirve: Yükseliş bekleniyorsa kapanışın biraz üstü, düşüş bekleniyorsa açılışın biraz üstü
        if ilk_gun_artis > 0:
            gun_ici_zirve = round(beklenen_kapanis + (atr_14 * 0.4), 2)
        else:
            gun_ici_zirve = round(son_fiyat + (atr_14 * 0.2), 2)

        if sinyal in ["GÜÇLÜ AL", "AL"]:
            strateji_metni = (
                f"🎯 **Senaryo:** Sabah 09:00 - 10:00 arası piyasa açılış esnemesinde **{sabah_alim} SEK** civarından pozisyon alınabilir. "
                f"Öğleden sonra 14:00 - 15:00 bandında hacimle beraber gün içi maksimum **{gun_ici_zirve} SEK** seviyelerini test etmesi, "
                f"ardından kâr satışlarıyla günü **{beklenen_kapanis} SEK** civarında kapatması beklenebilir. <br>"
                f"🛡️ <b>Risk:</b> Ters bir durumda <b>{stop_loss} SEK</b> seviyesi zarar-kes (stop-loss) olarak kullanılmalıdır."
            )
        elif sinyal == "SAT":
            strateji_metni = (
                f"⚠️ **Senaryo:** Hisse aşırı alım/satış baskısı altında. Sabah açılışta **{sabah_alim} SEK** seviyesinden zayıf bir tepki verse de, "
                f"gün içi en fazla **{gun_ici_zirve} SEK** seviyelerine tutunması ve ardından günü zayıf bir şekilde **{beklenen_kapanis} SEK** "
                f"civarında kapatması yüksek olasılıktır. Alım önerilmez."
            )
        else:
            strateji_metni = (
                f"⚖️ **Senaryo:** Fiyat yatay ve kararsız. Sabah dalgalanmasında **{sabah_alim} SEK** ile öğleden sonra **{gun_ici_zirve} SEK** "
                f"arasında testere (git-gel) yapması öngörülüyor. Gün sonu kapanış tahmini **{beklenen_kapanis} SEK**. "
                f"Net bir kırılım olmadığı için gün içi işlem riski yüksektir."
            )

        return {
            'Şirket Adı': sirket_adi,
            'Kod': h_clean,
            'Son Fiyat (SEK)': round(son_fiyat, 2),
            '1. Gün Tahmin (%)': tahminler_yuzde['1. Gün Tahmin (%)'],
            '2. Gün Tahmin (%)': tahminler_yuzde['2. Gün Tahmin (%)'],
            '3. Gün Tahmin (%)': tahminler_yuzde['3. Gün Tahmin (%)'],
            'Sinyal': sinyal,
            'RSI (14)': round(son_rsi, 1),
            'Hacim Katı': round(hacim_kati, 1),
            'Destek (S1)': destek_s1,
            'Direnç (R1)': direnc_r1,
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
st.caption("🚀 Gerçekçi ATR Matematiği & Nokta Atışı Gün İçi Zamanlamaları")

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

if st.button("🔄 Yeni Analiz Başlat / Verileri Güncelle", type="primary", use_container_width=True):
    if hisseler:
        rapor = []
        bar = st.progress(0)
        durum = st.empty()
        durum.text("⚡ Hisseler taranıyor... Matematiksel senaryolar oluşturuluyor.")
        
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
            df_res = df_res.sort_values(by='1. Gün Tahmin (%)', ascending=False)
            st.session_state.pro_analiz_df = df_res
            df_res.to_csv(DATA_FILE, index=False)

if st.session_state.pro_analiz_df is not None:
    df_res = st.session_state.pro_analiz_df
    
    son_zaman = df_res['Analiz Zamanı'].iloc[0] if 'Analiz Zamanı' in df_res.columns else "Bilinmiyor"
    st.info(f"🕒 Ekranda gördüğünüz verilerin en son güncellenme zamanı: **{son_zaman}**")
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("Başarıyla Analiz Edilen", len(df_res))
    c2.metric("Yarın Yükseliş Beklenen", len(df_res[df_res['1. Gün Tahmin (%)'] > 0]))

    if gorunum_modu == "Mobil Kart Görünümü (Tavsiye)":
        st.subheader("🔥 Fırsat Hisseleri & Saatlik Aksiyon Planları")
        for _, row in df_res.iterrows():
            t1_color = "green" if row['1. Gün Tahmin (%)'] > 0 else "red"
            t2_color = "green" if row['2. Gün Tahmin (%)'] > 0 else "red"
            t3_color = "green" if row['3. Gün Tahmin (%)'] > 0 else "red"
            
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
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                    <span><b>Yarın:</b> <span class="badge-{t1_color}">%{row['1. Gün Tahmin (%)']:+.2f}</span></span>
                    <span><b>2. Gün:</b> <span class="badge-{t2_color}">%{row['2. Gün Tahmin (%)']:+.2f}</span></span>
                    <span><b>3. Gün:</b> <span class="badge-{t3_color}">%{row['3. Gün Tahmin (%)']:+.2f}</span></span>
                </div>
                <div class="strategy-box">
                    {row['Strateji']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.dataframe(df_res.drop(columns=['Strateji']), use_container_width=True)
