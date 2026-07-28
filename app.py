import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import io

st.set_page_config(page_title="Avanza Detaylı Borsa Analiz & Tahmin", page_icon="📊", layout="wide")

def rsi_hesapla(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

st.title("🇸🇪 İsveç Borsası Çoklu Zaman Dilimi Analizi & 3 Günlük Tahmin Paneli")
st.caption("Günlük Net Yüzde Tahminleri (Al-Sat Değerlendirmesi İçin) + 3-1-1 Zaman Dilimi Analizi ve Min/Max Fiyat Tahminleri")

st.sidebar.header("⚙️ Ayarlar & Hisse Seçimi")
yukleme_yontemi = st.sidebar.radio("Hisse Giriş Yöntemi:", ["Varsayılan Tam Liste (114 Hisse)", "Excel Dosyası Yükle"])

# 114 Hisselik Tam İsveç Listesi
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

if yukleme_yontemi == "Varsayılan Tam Liste (114 Hisse)":
    girdi = st.sidebar.text_area("Hisse Listeniz:", value=varsayilan_liste, height=180)
    hisseler = [h.strip() for h in girdi.split(",") if h.strip()]
else:
    file = st.sidebar.file_uploader("Excel Yükle (.xlsx)", type=["xlsx"])
    if file:
        df_ex = pd.read_excel(file)
        if 'Hisse' in df_ex.columns:
            hisseler = df_ex['Hisse'].dropna().astype(str).tolist()

if st.sidebar.button("🚀 Detaylı Analiz & 3 Günlük Tahmin Başlat", type="primary"):
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
            durum_metni.text(f"Analiz ediliyor ({idx + 1}/{toplam_hisse}): {h_clean}...")
            
            try:
                ticker = yf.Ticker(h_yf)
                df = ticker.history(period="6mo", interval="1d")
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                df = df.dropna(subset=['Close', 'High', 'Low'])
                
                if df.empty or len(df) < 30:
                    continue

                sirket_adi = h_clean
                try:
                    info = ticker.info
                    sirket_adi = info.get('longName') or info.get('shortName') or h_clean
                except:
                    sirket_adi = h_clean

                close = df['Close'].dropna()
                high = df['High'].dropna()
                low = df['Low'].dropna()
                
                if close.empty:
                    continue

                son_fiyat = float(close.iloc[-1])
                son_high = float(high.iloc[-1])
                son_low = float(low.iloc[-1])

                # --- ÇOKLU ZAMAN DİLİMİ PERFORMANS & VOLATİLİTE ---
                df_1w = close.tail(5)
                getiri_1w = ((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]) * 100 if len(close) >= 5 else 0.0
                vol_1w = df_1w.pct_change().std() * 100 if len(df_1w) > 1 else 0.0

                df_1m = close.tail(20)
                getiri_1m = ((close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]) * 100 if len(close) >= 20 else 0.0
                vol_1m = df_1m.pct_change().std() * 100 if len(df_1m) > 1 else 0.0

                df_3m = close.tail(60)
                getiri_3m = ((close.iloc[-1] - close.iloc[-60]) / close.iloc[-60]) * 100 if len(close) >= 60 else 0.0
                vol_3m = df_3m.pct_change().std() * 100 if len(df_3m) > 1 else 0.0

                # --- TEKNİK GÖSTERGELER ---
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

                # --- ÖNÜMÜZDEKİ 3 GÜN İÇİN NET YÜZDE VE MIN/MAX TAHMİNİ ---
                gunluk_getiriler = df_1m.pct_change().dropna()
                gunluk_ort_getiri = gunluk_getiriler.mean() * 100 if len(gunluk_getiriler) > 0 else 0.0
                gunluk_vol = gunluk_getiriler.std() * 100 if len(gunluk_getiriler) > 0 else 1.0

                sma_5 = close.tail(5).mean()
                sma_20 = close.tail(20).mean()
                
                if son_fiyat > sma_5 and sma_5 > sma_20:
                    yon = 1.0
                elif son_fiyat < sma_5 and sma_5 < sma_20:
                    yon = -1.0
                else:
                    yon = 0.2 if gunluk_ort_getiri > 0 else -0.2

                # 1., 2. ve 3. Gün Net Yüzde Tahminleri (Eski mantığa dayalı net tahmin)
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
                    
                    pct_min = round(((gun_min - son_fiyat) / son_fiyat) * 100, 2)
                    pct_max = round(((gun_max - son_fiyat) / son_fiyat) * 100, 2)
                    
                    tahminler_3gun[f'Gün {gun} Min'] = gun_min
                    tahminler_3gun[f'Gün {gun} Min %'] = pct_min
                    tahminler_3gun[f'Gün {gun} Max'] = gun_max
                    tahminler_3gun[f'Gün {gun} Max %'] = pct_max
                    
                    anlik_fiyat = gun_merkez

                rapor.append({
                    'Şirket Adı': sirket_adi,
                    'Kod': h_clean,
                    'Son Fiyat (SEK)': round(son_fiyat, 2),
                    
                    # SON FİYATTAN SONRA GELEN NET 3 GÜNLÜK TAHMİN SÜTUNLARI (AL-SAT DEĞERLENDİRMESİ İÇİN)
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
                    
                    # MIN/MAX Tahminleri
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

            except Exception as e:
                pass

        durum_metni.empty()

        if rapor:
            df_res = pd.DataFrame(rapor)
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Analiz Edilen Hisse", len(df_res))
            col2.metric("1. Gün Yükseliş Beklenenler", len(df_res[df_res['1. Gün Tahmin (%)'] > 0]))
            col3.metric("1. Gün Düşüş Beklenenler", len(df_res[df_res['1. Gün Tahmin (%)'] < 0]))

            st.subheader("📊 Çoklu Zaman Dilimi & Net 3 Günlük Tahmin Tablosu")
            
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
                    '1. Gün Min': '{:.2f}',
                    '1. Gün Min (%)': '{:+.2f}%',
                    '1. Gün Max': '{:.2f}',
                    '1. Gün Max (%)': '{:+.2f}%',
                    '2. Gün Min': '{:.2f}',
                    '2. Gün Min (%)': '{:+.2f}%',
                    '2. Gün Max': '{:.2f}',
                    '2. Gün Max (%)': '{:+.2f}%',
                    '3. Gün Min': '{:.2f}',
                    '3. Gün Min (%)': '{:+.2f}%',
                    '3. Gün Max': '{:.2f}',
                    '3. Gün Max (%)': '{:+.2f}%',
                    'Destek (S1)': '{:.2f}',
                    'Direnç (R1)': '{:.2f}',
                    'Stop-Loss': '{:.2f}',
                    'RSI (14)': '{:.1f}',
                    'Hacim Katı': '{:.1f}'
                }), 
                use_container_width=True
            )
            
            dosya_adi = f"isvec_3gunluk_al_sat_tahminleri_{bugun_tarih}.xlsx"

            # Excel çıktısını belleğe yazıp indirme
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_res.to_excel(writer, index=False, sheet_name='Analiz ve Tahminler')
            excel_data = output.getvalue()
            
            st.download_button(
                label=f"📥 Tabloyu Excel Dosyası Olarak İndir (.xlsx)",
                data=excel_data,
                file_name=dosya_adi,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
        else:
            st.error("Veriler çekilemedi. Lütfen internet bağlantınızı kontrol edin.")