import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Avanza Detaylı Borsa Analiz", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Mobil Özel CSS Stilleri
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

# Yahoo Direct API üzerinden veri çekme fonksiyonu (Bot engelini aşar)
def yahoo_veri_cek(symbol):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=6mo&interval=1d"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Origin': 'https://finance.yahoo.com',
        'Referer': f'https://finance.yahoo.com/quote/{symbol}'
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            return None
        
        data = res.json()
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        quote = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'Open': quote.get('open'),
            'High': quote.get('high'),
            'Low': quote.get('low'),
            'Close': quote.get('close'),
            'Volume': quote.get('volume')
        }, index=pd.to_datetime(timestamps, unit='s'))
        
        df = df.dropna(subset=['Close', 'High', 'Low'])
        return df
    except Exception:
        return None

def rsi_hesapla(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

st.title("🇸🇪 İsveç Borsası Analiz & Tahmin")
st.caption("📱 Mobil & Masaüstü Uyumlu Al-Sat Tahmin Paneli")

# --- YAN PANEL (SETTINGS) ---
st.sidebar.header("⚙️ Ayarlar & Filtreler")

gorunum_modu = st.sidebar.radio(
    "📱 Görünüm Modu:", 
    ["Mobil Kart Görünümü (Tavsiye)", "Klasik Masaüstü Tablosu"]
)

yukleme_yontemi = st.sidebar.radio("Hisse Giriş Yöntemi:", ["Varsayılan Liste (114 Hisse)", "Excel / CSV Yükle"])

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

hisseler = []

if yukleme_yontemi == "Varsayılan Liste (114 Hisse)":
    girdi = st.sidebar.text_area("Hisse Listeniz:", value=varsayilan_liste, height=120)
    hisseler = [h.strip() for h in girdi.split(",") if h.strip()]
else:
    file = st.sidebar.file_uploader("Dosya Yükle (.xlsx veya .csv)", type=["xlsx", "csv"])
    if file:
        try:
            if file.name.endswith('.csv'):
                df_ex = pd.read_csv(file)
            else:
                df_ex = pd.read_excel(file)
            if 'Hisse' in df_ex.columns:
                hisseler = df_ex['Hisse'].dropna().astype(str).tolist()
        except Exception:
            st.sidebar.error("Dosya okunamadı. Lütfen CSV formatında deneyin.")

sinyal_filtre = st.sidebar.multiselect(
    "Sinyal Filtresi:",
    ["GÜÇLÜ AL", "AL", "NÖTR", "SAT"],
    default=["GÜÇLÜ AL", "AL", "NÖTR", "SAT"]
)

# Mobil Uyumlu Analiz Butonu
if st.button("🚀 Analizi Başlat", type="primary", use_container_width=True):
    if not hisseler:
        st.warning("Lütfen analiz etmek için hisse ekleyin.")
    else:
        rapor = []
        bar = st.progress(0)
        durum_metni = st.empty()
        
        toplam_hisse = len(hisseler)
        bugun_tarih = datetime.now().strftime("%Y-%m-%d")
        
        for idx, h_kod in enumerate(hisseler):
            h_clean = str(h_kod).strip().upper()
            h_yf = h_clean if h_clean.endswith(".ST") else f"{h_clean}.ST"
            
            bar.progress((idx + 1) / toplam_hisse)
            durum_metni.text(f"Analiz ediliyor ({idx + 1}/{toplam_hisse}): {h_clean}")
            
            try:
                df = yahoo_veri_cek(h_yf)
                if df is None or df.empty or len(df) < 30:
                    continue

                sirket_adi = h_clean

                close = df['Close']
                high = df['High']
                low = df['Low']
                
                son_fiyat = float(close.iloc[-1])
                son_high = float(high.iloc[-1])
                son_low = float(low.iloc[-1])

                # Getiri & Volatilite
                df_1w = close.tail(5)
                getiri_1w = ((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]) * 100 if len(close) >= 5 else 0.0
                vol_1w = df_1w.pct_change().std() * 100 if len(df_1w) > 1 else 0.0

                df_1m = close.tail(20)
                getiri_1m = ((close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]) * 100 if len(close) >= 20 else 0.0
                vol_1m = df_1m.pct_change().std() * 100 if len(df_1m) > 1 else 0.0

                df_3m = close.tail(60)
                getiri_3m = ((close.iloc[-1] - close.iloc[-60]) / close.iloc[-60]) * 100 if len(close) >= 60 else 0.0
                vol_3m = df_3m.pct_change().std() * 100 if len(df_3m) > 1 else 0.0

                # Göstergeler
                rsi_series = rsi_hesapla(close)
                rsi_valid = rsi_series.dropna()
                son_rsi = float(rsi_valid.iloc[-1]) if not rsi_valid.empty else 50.0

                pivot = (son_high + son_low + son_fiyat) / 3.0
                destek_s1 = (2.0 * pivot) - son_high
                direnc_r1 = (2.0 * pivot) - son_low
                stop_loss = son_fiyat * 0.95

                if 'Volume' in df.columns and not df['Volume'].dropna().empty:
                    vol_s = df['Volume'].dropna()
                    ort_hacim = vol_s.tail(10).mean()
                    son_hacim = vol_s.iloc[-1]
                    hacim_kati = (son_hacim / ort_hacim) if ort_hacim > 0 else 1.0
                else:
                    hacim_kati = 1.0

                if son_rsi < 30 and hacim_kati > 1.2:
                    sinyal = "GÜÇLÜ AL"
                elif son_rsi < 40:
                    sinyal = "AL"
                elif son_rsi > 70:
                    sinyal = "SAT"
                else:
                    sinyal = "NÖTR"

                # 3 Günlük Yüzde Tahminleri
                gunluk_getiriler = df_1m.pct_change().dropna()
                gunluk_ort_getiri = gunluk_getiriler.mean() * 100 if len(gunluk_getiriler) > 0 else 0.0
                gunluk_vol = gunluk_getiriler.std() * 100 if len(gunluk_getiriler) > 0 else 1.0

                sma_5 = close.tail(5).mean()
                sma_20 = close.tail(20).mean()
                
                yon = 1.0 if son_fiyat > sma_5 > sma_20 else (-1.0 if son_fiyat < sma_5 < sma_20 else (0.2 if gunluk_ort_getiri > 0 else -0.2))

                tahmin_yuzde_1 = gunluk_ort_getiri + (yon * (gunluk_vol * 0.4))
                tahmin_yuzde_2 = gunluk_ort_getiri + (yon * (gunluk_vol * 0.3))
                tahmin_yuzde_3 = gunluk_ort_getiri + (yon * (gunluk_vol * 0.2))

                tahminler_3gun = {}
                anlik_fiyat = son_fiyat
                for gun in range(1, 4):
                    gun_merkez = anlik_fiyat * (1.0 + (gun * (gunluk_ort_getiri/100) * yon))
                    gun_sapma = son_fiyat * (gunluk_vol/100) * 1.2 * (1 + (gun * 0.1))
                    
                    gun_min = round(max(gun_merkez - gun_sapma, son_fiyat * 0.5), 2)
                    gun_max = round(gun_merkez + gun_sapma, 2)
                    
                    tahminler_3gun[f'Gün {gun} Min'] = gun_min
                    tahminler_3gun[f'Gün {gun} Min %'] = round(((gun_min - son_fiyat) / son_fiyat) * 100, 2)
                    tahminler_3gun[f'Gün {gun} Max'] = gun_max
                    tahminler_3gun[f'Gün {gun} Max %'] = round(((gun_max - son_fiyat) / son_fiyat) * 100, 2)
                    anlik_fiyat = gun_merkez

                rapor.append({
                    'Şirket Adı': sirket_adi,
                    'Kod': h_clean,
                    'Son Fiyat (SEK)': round(son_fiyat, 2),
                    '1. Gün Tahmin (%)': round(tahmin_yuzde_1, 2),
                    '2. Gün Tahmin (%)': round(tahmin_yuzde_2, 2),
                    '3. Gün Tahmin (%)': round(tahmin_yuzde_3, 2),
                    'Sinyal': sinyal,
                    '3 Aylık Getiri (%)': round(getiri_3m, 2),
                    '1 Aylık Getiri (%)': round(getiri_1m, 2),
                    '1 Haftalık Getiri (%)': round(getiri_1w, 2),
                    '3 Aylık Volatilite (%)': round(vol_3m, 2),
                    '1 Aylık Volatilite (%)': round(vol_1m, 2),
                    '1 Haftalık Volatilite (%)': round(vol_1w, 2),
                    '1. Gün Min': tahminler_3gun['Gün 1 Min'],
                    '1. Gün Min (%)': tahminler_3gun['Gün 1 Min %'],
                    '1. Gün Max': tahminler_3gun['Gün 1 Max'],
                    '1. Gün Max (%)': tahminler_3gun['Gün 1 Max %'],
                    '2. Gün Min': tahminler_3gun['Gün 2 Min'],
                    '2. Gün Min (%)': tahminler_3gun['Gün 2 Min %'],
                    '2. Gün Max': tahminler_3gun['Gün 2 Max'],
                    '2. Gün Max (%)': tahminler_3gun['Gün 2 Max %'],
                    '3. Gün Min': tahminler_3gun['Gün 3 Min'],
                    '3. Gün Min (%)': tahminler_3gun['Gün 3 Min %'],
                    '3. Gün Max': tahminler_3gun['Gün 3 Max'],
                    '3. Gün Max (%)': tahminler_3gun['Gün 3 Max %'],
                    'Destek (S1)': round(destek_s1, 2),
                    'Direnç (R1)': round(direnc_r1, 2),
                    'Stop-Loss': round(stop_loss, 2),
                    'RSI (14)': round(son_rsi, 1),
                    'Hacim Katı': round(float(hacim_kati), 1),
                    'Analiz Tarihi': bugun_tarih
                })

            except Exception:
                pass

        durum_metni.empty()

        if rapor:
            df_res = pd.DataFrame(rapor)
            
            # Filtreleme
            if sinyal_filtre:
                df_res = df_res[df_res['Sinyal'].isin(sinyal_filtre)]

            st.markdown("---")
            # Mobilde Özet Metrikler
            c1, c2 = st.columns(2)
            c1.metric("Analiz Edilen", len(df_res))
            c2.metric("1. Gün Yükseliş Beklenen", len(df_res[df_res['1. Gün Tahmin (%)'] > 0]))

            # MOBİL KART GÖRÜNÜMÜ
            if gorunum_modu == "Mobil Kart Görünümü (Tavsiye)":
                st.subheader("📱 Mobil Hisse Kartları")
                for _, row in df_res.iterrows():
                    t1_color = "green" if row['1. Gün Tahmin (%)'] > 0 else "red"
                    t2_color = "green" if row['2. Gün Tahmin (%)'] > 0 else "red"
                    t3_color = "green" if row['3. Gün Tahmin (%)'] > 0 else "red"
                    
                    st.markdown(f"""
                    <div class="stock-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h4 style="margin:0; font-size: 1.1rem;">{row['Kod']} - <span style="font-size:0.9rem; font-weight:normal;">{row['Şirket Adı'][:18]}</span></h4>
                            <span style="background-color: {'#d4edda' if row['Sinyal'] in ['AL', 'GÜÇLÜ AL'] else '#f8d7da'}; color: {'#155724' if row['Sinyal'] in ['AL', 'GÜÇLÜ AL'] else '#721c24'}; padding: 3px 8px; border-radius: 5px; font-weight: bold; font-size: 0.8rem;">{row['Sinyal']}</span>
                        </div>
                        <div style="margin-top: 8px; font-size: 0.95rem;">
                            <b>Fiyat:</b> {row['Son Fiyat (SEK)']} SEK | <b>RSI:</b> {row['RSI (14)']} | <b>Hacim:</b> {row['Hacim Katı']}x
                        </div>
                        <hr style="margin: 8px 0;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                            <span><b>1. Gün:</b> <span class="badge-{t1_color}">%{row['1. Gün Tahmin (%)']:+}</span></span>
                            <span><b>2. Gün:</b> <span class="badge-{t2_color}">%{row['2. Gün Tahmin (%)']:+}</span></span>
                            <span><b>3. Gün:</b> <span class="badge-{t3_color}">%{row['3. Gün Tahmin (%)']:+}</span></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander(f"🔍 {row['Kod']} Detaylı Min/Max & Destek"):
                        st.write(f"**1. Gün Aralık:** {row['1. Gün Min']} ({row['1. Gün Min (%)']}%) — {row['1. Gün Max']} ({row['1. Gün Max (%)']}%)")
                        st.write(f"**Destek (S1):** {row['Destek (S1)']} | **Direnç (R1):** {row['Direnç (R1)']}")
                        st.write(f"**Stop-Loss:** {row['Stop-Loss']}")

            else:
                # MASAÜSTÜ TABLOSU
                st.subheader("📊 Detaylı Analiz Tablosu")
                st.dataframe(
                    df_res.style.format({
                        'Son Fiyat (SEK)': '{:.2f}',
                        '1. Gün Tahmin (%)': '{:+.2f}%',
                        '2. Gün Tahmin (%)': '{:+.2f}%',
                        '3. Gün Tahmin (%)': '{:+.2f}%',
                        '3 Aylık Getiri (%)': '{:.2f}',
                        '1 Aylık Getiri (%)': '{:.2f}',
                        '1 Haftalık Getiri (%)': '{:.2f}',
                        '3 Aylık Volatilite (%)': '{:.2f}',
                        '1 Aylık Volatilite (%)': '{:.2f}',
                        '1 Haftalık Volatilite (%)': '{:.2f}',
                        'RSI (14)': '{:.1f}',
                        'Hacim Katı': '{:.1f}'
                    }), 
                    use_container_width=True
                )

            # CSV İndirme
            csv_data = df_res.to_csv(index=False).encode('utf-8-sig')
            
            st.download_button(
                label="📥 Tüm Sonuçları İndir (.csv)",
                data=csv_data,
                file_name=f"isvec_borsa_analiz_{bugun_tarih}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
        else:
            st.error("Veriler çekilemedi.")
