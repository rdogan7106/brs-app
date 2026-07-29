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
    .strategy-box {
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        font-size: 0.9rem;
        color: #0d47a1;
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
# PRO FONKSİYONLAR: PARALEL VERİ ÇEKME & TEKNİK ANALİZ ENGINE
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
        son_yuksek = float(high.iloc[-1])
        son_dusuk = float(low.iloc[-1])
        
        # PRO İNDİKATÖRLER
        ema_5 = close.ewm(span=5, adjust=False).mean().iloc[-1]
        ema_20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        son_rsi = float((100 - (100 / (1 + rs))).iloc[-1])

        vol_s = df['Volume'].dropna()
        ort_hacim = vol_s.tail(10).mean() if len(vol_s) >= 10 else 1
        hacim_kati = float(vol_s.iloc[-1] / ort_hacim) if ort_hacim > 0 else 1.0

        if hacim_kati > 1.4 and son_fiyat > ema_5 and ema_5 > ema_20 and 48 <= son_rsi <= 68:
            sinyal = "GÜÇLÜ AL"
        elif (son_rsi < 35 and hacim_kati > 1.2) or (son_fiyat > ema_5 and son_rsi < 62):
            sinyal = "AL"
        elif son_rsi > 70 or (son_fiyat < ema_5 and hacim_kati > 1.5):
            sinyal = "SAT"
        else:
            sinyal = "NÖTR"

        gunluk_getiri = close.pct_change().dropna()
        volatilite = gunluk_getiri.tail(20).std() * 100
        ort_getiri = gunluk_getiri.tail(20).mean() * 100
        
        yon = 1.0 if son_fiyat > ema_5 > ema_20 else (-1.0 if son_fiyat < ema_5 else 0.2)
        
        # Pivot & Seviyeler
        pivot = (son_yuksek + son_dusuk + son_fiyat) / 3.0
        destek_s1 = round((2.0 * pivot) - son_yuksek, 2)
        direnc_r1 = round((2.0 * pivot) - son_dusuk, 2)
        direnc_r2 = round(pivot + (son_yuksek - son_dusuk), 2)
        stop_loss = round(son_fiyat * 0.982, 2)

        # 3 GÜNLÜK TAHMİN (Geri Eklendi)
        tahminler_yuzde = {}
        anlik_fiyat = son_fiyat
        for gun in range(1, 4):
            # İvme ve hacim faktörünü günlere yayıyoruz
            gun_artis = ort_getiri + (yon * (volatilite * 0.35)) + ((hacim_kati - 1.0) * (0.6 / gun))
            anlik_fiyat = anlik_fiyat * (1 + (gun_artis / 100))
            tahminler_yuzde[f'{gun}. Gün Tahmin (%)'] = round(((anlik_fiyat - son_fiyat) / son_fiyat) * 100, 2)

        # YAPAY ZEKA STİLİ AL-SAT STRATEJİ METNİ
        if sinyal in ["GÜÇLÜ AL", "AL"]:
            saat_vurgusu = "14:00'ten sonra kâr satışları (afternoon fade) gelme ihtimali yüksek" if hacim_kati > 1.5 else "Öğleden sonra piyasa yönüne dikkat edilmeli"
            alim_yeri = max(destek_s1, round(son_fiyat * 0.99, 2))
            
            strateji_metni = (
                f"🎯 **Tavsiye:** Sabah seans açılışında muhtemel dalgalanmayı bekleyip, hisseyi **{alim_yeri} SEK** civarından (Destek/S1) yakalamaya çalışın. "
                f"Hacim {hacim_kati:.1f}x arttığı için sabah saatlerinde hızlı bir atakla **{direnc_r1} - {direnc_r2} SEK** bandına (R1/R2) ulaşabilir. "
                f"💡 **Kritik Saat:** {saat_vurgusu}, bu yüzden hedef fiyata ulaştığında kârı realize edip çıkmak en güvenlisidir. "
                f"Olası ters durumda **{stop_loss} SEK** altında kesinlikle pozisyonu kapatın."
            )
        elif sinyal == "SAT":
            strateji_metni = (
                f"⚠️ **Tavsiye:** Hisse RSI ({son_rsi:.1f}) seviyesinde ve satış baskısı yiyor. "
                f"Gün içi alım (long) yapmak risklidir. İzlemede kalınması tavsiye edilir."
            )
        else:
            strateji_metni = (
                f"⚖️ **Tavsiye:** Fiyat yatay seyrediyor. Eğer **{direnc_r1} SEK** hacimli kırılırsa alım denenebilir. "
                f"Kırılmazsa gün içi testere piyasasına kurban olmamak için işlemden uzak durulmalıdır."
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
            'Analiz Tarihi': datetime.now().strftime("%Y-%m-%d")
        }
    except Exception:
        return None

# ---------------------------------------------------------
# UI & STREAMLIT ARAYÜZÜ
# ---------------------------------------------------------

st.title("⚡ Avanza Pro Day-Trading Analizi")
st.caption("🚀 Çoklu İşlemci + 3 Günlük Projeksiyon + Akıllı Trade Tavsiyeleri")

st.sidebar.header("⚙️ Ayarlar")
gorunum_modu = st.sidebar.radio("📱 Görünüm Modu:", ["Mobil Kart Görünümü (Tavsiye)", "Klasik Masaüstü Tablosu"])

varsayilan_liste = (
    "ELUX-B, ANOD-B, LIME, ELUX-A, MIPS, ARLA, NTEK-B, PRIC-B, BILL, BOOZT, NANO, QLINEA, XBRANE, HOLM-B, "
    "PRFO, RAY-B, BESQ-B, PROFF, CLAS-B, REJL-B, SEZI, SECT-B, BIOA-B, SKIS-B, RROS, CER, YUBICO, HOLM-A, "
    "KABE-B, CARA, ESSITY-A, SOBI, EPEN, ESSITY-B, PREV-B, CEVI, HMS, HEXA-B, SWEC-A, THULE, APOTEA, TROAX"
)

girdi = st.sidebar.text_area("Hisse Listeniz (Örnek Kısa Liste):", value=varsayilan_liste, height=120)
hisseler = [h.strip() for h in girdi.split(",") if h.strip()]

if "pro_analiz_df" not in st.session_state:
    st.session_state.pro_analiz_df = None

if st.button("🚀 Hızlı Analizi Başlat (Pro Engine)", type="primary", use_container_width=True):
    if hisseler:
        rapor = []
        bar = st.progress(0)
        durum = st.empty()
        durum.text("⚡ Paralel taranıyor... Lütfen bekleyin.")
        
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
            df_res = df_res.sort_values(by='1. Gün Tahmin (%)', ascending=False)
            st.session_state.pro_analiz_df = df_res

if st.session_state.pro_analiz_df is not None:
    df_res = st.session_state.pro_analiz_df
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("Analiz Edilen Hisse", len(df_res))
    c2.metric("Yarın Yükseliş Beklenen", len(df_res[df_res['1. Gün Tahmin (%)'] > 0]))

    if gorunum_modu == "Mobil Kart Görünümü (Tavsiye)":
        st.subheader("🔥 Fırsat Hisseleri & Aksiyon Planları")
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
        # Tablo görünümünde karmaşık metni gizleyip temel verileri gösteriyoruz
        st.dataframe(df_res.drop(columns=['Strateji']), use_container_width=True)
