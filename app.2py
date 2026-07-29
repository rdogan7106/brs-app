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
    page_title="Avanza Pro Hybrid Panel", 
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
    .buy-box {
        background-color: #e8f5e9;
        border-left: 5px solid #2e7d32;
        padding: 8px;
        border-radius: 5px;
        margin-top: 8px;
        font-size: 0.9rem;
        color: #1b5e20;
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
        .buy-box {
            background-color: #1b281d;
            border-left: 5px solid #81c784;
            color: #c8e6c9;
        }
    }
    .badge-green { color: #2e7d32; font-weight: bold; }
    .badge-red { color: #c62828; font-weight: bold; }
    .badge-neutral { color: #f57f17; font-weight: bold; }
    .divider { border-top: 1px dashed #ccc; margin: 10px 0; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# PRO FONKSİYONLAR: DUAL-FETCH (GÜN İÇİ + GÜNLÜK VERİ)
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
    
    url_15m = f"https://query1.finance.yahoo.com/v8/finance/chart/{h_yf}?range=5d&interval=15m"
    url_1d = f"https://query1.finance.yahoo.com/v8/finance/chart/{h_yf}?range=3mo&interval=1d"
    
    try:
        res_15m = session.get(url_15m, timeout=6)
        res_1d = session.get(url_1d, timeout=6)
        
        if res_15m.status_code != 200 or res_1d.status_code != 200:
            return None
        
        # --- 15 DAKİKALIK VERİ İŞLEME ---
        data_15m = res_15m.json()['chart']['result'][0]
        meta = data_15m.get('meta', {})
        sirket_adi = meta.get('shortName') or meta.get('longName') or h_clean
        
        df_15m = pd.DataFrame({
            'Close': data_15m['indicators']['quote'][0].get('close'),
            'High': data_15m['indicators']['quote'][0].get('high'),
            'Low': data_15m['indicators']['quote'][0].get('low'),
            'Volume': data_15m['indicators']['quote'][0].get('volume')
        }).dropna()

        if len(df_15m) < 30: return None

        son_fiyat = float(df_15m['Close'].iloc[-1])
        
        sma_20 = df_15m['Close'].rolling(window=20).mean()
        std_20 = df_15m['Close'].rolling(window=20).std()
        upper_band = (sma_20 + (std_20 * 2)).iloc[-1]
        lower_band = (sma_20 - (std_20 * 2)).iloc[-1]
        
        vwap = (df_15m['Close'] * df_15m['Volume']).rolling(14).sum() / df_15m['Volume'].rolling(14).sum()
        son_vwap = round(vwap.iloc[-1], 2)
        
        delta = df_15m['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        son_rsi_15m = float((100 - (100 / (1 + (gain / loss)))).iloc[-1])

        # --- GÜNLÜK VERİ İŞLEME ---
        data_1d = res_1d.json()['chart']['result'][0]
        df_1d = pd.DataFrame({
            'Close': data_1d['indicators']['quote'][0].get('close'),
            'High': data_1d['indicators']['quote'][0].get('high'),
            'Low': data_1d['indicators']['quote'][0].get('low'),
            'Volume': data_1d['indicators']['quote'][0].get('volume')
        }).dropna()
        
        close_1d = df_1d['Close']
        
        high_low = df_1d['High'] - df_1d['Low']
        high_close = (df_1d['High'] - close_1d.shift(1)).abs()
        low_close = (df_1d['Low'] - close_1d.shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_14_1d = float(true_range.rolling(14).mean().iloc[-1])
        atr_yuzde = (atr_14_1d / son_fiyat) * 100

        delta_1d = close_1d.diff()
        rs_1d = (delta_1d.where(delta_1d > 0, 0)).rolling(14).mean() / (-delta_1d.where(delta_1d < 0, 0)).rolling(14).mean()
        son_rsi_1d = float((100 - (100 / (1 + rs_1d))).iloc[-1])

        gunluk_getiri = close_1d.pct_change().dropna()
        ema_getiri = gunluk_getiri.ewm(span=10, adjust=False).mean().iloc[-1] * 100

        if son_rsi_1d > 75: momentum = -0.15 
        elif son_rsi_1d < 30: momentum = 0.15  
        else: momentum = (son_rsi_1d - 50) / 100.0

        vol_s_1d = df_1d['Volume'].tail(10)
        hacim_kati = float(vol_s_1d.iloc[-1] / vol_s_1d.mean()) if vol_s_1d.mean() > 0 else 1.0
        hacim_etkisi = min(hacim_kati, 2.0)

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

        yarin_alis = round(son_fiyat - (atr_14_1d * 0.35), 2)
        potansiyel_pik = round(upper_band + (atr_14_1d * 0.1), 2)

        if son_fiyat >= upper_band and son_rsi_15m > 70:
            sinyal = "PİK YAPTI (SAT)"
            strateji_metni = (
                f"🚨 **DİKKAT (ZİRVE):** Fiyat Bollinger üst bandını aştı. Geri çekilme riski çok yüksek. "
                f"Kârı cebe yakışır prensibiyle satıp nakde geçme zamanı."
            )
            saatlik_yon = -1.0 
        elif hacim_kati > 1.2 and son_fiyat > son_vwap and tahminler_yuzde['1. Gün Tahmin (%)'] > 0:
            sinyal = "GÜÇLÜ AL & TUT"
            strateji_metni = (
                f"🔥 **MOMENTUM YÜKSEK:** Fiyat VWAP'ın ({son_vwap} SEK) üzerinde. Yükseliş trendi güçlü. "
                f"Gün içi potansiyel zirve hedefi **{potansiyel_pik} SEK** seviyesidir."
            )
            saatlik_yon = 1.5
        elif son_fiyat < lower_band and son_rsi_15m < 30:
            sinyal = "DİPTEN TEPKİ (AL)"
            strateji_metni = (
                f"🎯 **DİP FIRSATI:** Hisse gün içi aşırı satıldı. "
                f"Kısa vadeli bir işlemle **{son_vwap} SEK** seviyesine kadar tepki sıçraması beklenebilir."
            )
            saatlik_yon = 1.0
        else:
            sinyal = "NÖTR (İZLE)"
            strateji_metni = (
                f"⚖️ **YATAY SEYİR:** Net bir trend yok. Yukarı kırılımda {round(upper_band, 2)} SEK, "
                f"aşağı kırılımda ise {round(lower_band, 2)} SEK hedeflenebilir."
            )
            saatlik_yon = 0.0

        beklenen_1h_getiri = round((atr_14_1d / son_fiyat / 4) * 100 * saatlik_yon, 2)

        return {
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
    except Exception:
        return None

# ---------------------------------------------------------
# UI & STREAMLIT ARAYÜZÜ
# ---------------------------------------------------------
st.title("⚡ Avanza Pro Hybrid Analiz")
st.caption("🚀 15-Dakikalık Zirve Avcısı + Günlük Çoklu Tahmin Motoru")

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
            df_res = df_res.sort_values(by=['1. Gün Tahmin (%)', '1 Saatlik Yön (%)'], ascending=[False, False])
            st.session_state.pro_analiz_df = df_res
            df_res.to_csv(DATA_FILE, index=False)

if st.session_state.pro_analiz_df is not None:
    df_res = st.session_state.pro_analiz_df
    
    son_zaman = df_res['Analiz Zamanı'].iloc[0] if 'Analiz Zamanı' in df_res.columns else "Bilinmiyor"
    st.info(f"🕒 Ekranda gördüğünüz verilerin en son güncellenme zamanı: **{son_zaman}**")
    
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Başarıyla Analiz Edilen", len(df_res))
    c2.metric("Gün İçi (Day-Trade) İvmeli", len(df_res[df_res['1 Saatlik Yön (%)'] > 0]))
    c3.metric("Yarın Yükseliş Beklenen", len(df_res[df_res['1. Gün Tahmin (%)'] > 0]))

    if gorunum_modu == "Mobil Kart Görünümü (Tavsiye)":
        for _, row in df_res.iterrows():
            t1_color = "green" if row['1 Saatlik Yön (%)'] > 0 else "red" if row['1 Saatlik Yön (%)'] < 0 else "neutral"
            d1_color = "green" if row['1. Gün Tahmin (%)'] > 0 else "red"
            d2_color = "green" if row['2. Gün Tahmin (%)'] > 0 else "red"
            d3_color = "green" if row['3. Gün Tahmin (%)'] > 0 else "red"

            if "AL" in row['Sinyal']: bg_color, text_color = '#d4edda', '#155724'
            elif "SAT" in row['Sinyal']: bg_color, text_color = '#f8d7da', '#721c24'
            else: bg_color, text_color = '#fff3cd', '#856404'
            
            # BURADAKİ EKSİK PARAMETRE (unsafe_allow_html=True) EKLENDİ:
            st.markdown(f"""
            <div class="stock-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin:0; font-size: 1.05rem;">{row['Şirket Adı']} <span style="font-size:0.85rem; color:#6c757d;">({row['Kod']})</span></h4>
                    <span style="background-color: {bg_color}; color: {text_color}; padding: 3px 8px; border-radius: 5px; font-weight: bold; font-size: 0.8rem;">{row['Sinyal']}</span>
                </div>
                <div style="margin-top: 8px; font-size: 0.9rem;">
                    <b>Fiyat:</b> {row['Son Fiyat (SEK)']} SEK | <b>RSI(15m):</b> {row['RSI (15m)']}
                </div>
                
                <div class="divider"></div>
                
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 5px;">
                    <span><b>Day-Trade (1 Saat):</b> <span class="badge-{t1_color}">%{row['1 Saatlik Yön (%)']:+.2f}</span></span>
                    <span><b>VWAP:</b> {row['VWAP']}</span>
                    <span><b>Pik Hedefi:</b> <span class="badge-green">{row['Potansiyel Pik']}</span></span>
                </div>
                
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                    <span><b>1. Gün:</b> <span class="badge-{d1_color}">%{row['1. Gün Tahmin (%)']:+.2f}</span></span>
                    <span><b>2. Gün:</b> <span class="badge-{d2_color}">%{row['2. Gün Tahmin (%)']:+.2f}</span></span>
                    <span><b>3. Gün:</b> <span class="badge-{d3_color}">%{row['3. Gün Tahmin (%)']:+.2f}</span></span>
                </div>
                
                <div class="strategy-box">
                    {row['Strateji']}
                </div>
                <div class="buy-box">
                    📅 <b>Yarınki Plan:</b> Olası bir sabah esnemesinde <b>{row['Yarınki Alış']} SEK</b> seviyesi, yeni pozisyon açmak veya maliyet düşürmek için güvenli bir alım noktasıdır.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.dataframe(df_res.drop(columns=['Strateji']), use_container_width=True)
