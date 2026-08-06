import requests
import pandas as pd
from datetime import datetime
from indicators import compute_rsi, compute_atr

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0'})

def fetch_yahoo_data(h_kod, interval="1d", range_param="3mo", include_vix=False):
    h_clean = str(h_kod).strip().upper()
    h_yf = h_clean if h_clean.endswith(".ST") else f"{h_clean}.ST"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{h_yf}?range={range_param}&interval={interval}"
    try:
        res = session.get(url, timeout=8)
        if res.status_code != 200: return None, None
        data = res.json()['chart']['result'][0]
        meta = data.get('meta', {})
        sirket_adi = meta.get('shortName') or meta.get('longName') or h_clean
        timestamps = data.get('timestamp', [])
        if not timestamps: return None, None
        quote_data = data['indicators']['quote'][0]
        df = pd.DataFrame({'Open': quote_data.get('open'), 'Close': quote_data.get('close'), 'High': quote_data.get('high'), 'Low': quote_data.get('low'), 'Volume': quote_data.get('volume')}, index=pd.to_datetime(timestamps, unit='s')).dropna()
        
        if include_vix and interval == "1d":
            try:
                vix_res = session.get(f"https://query1.finance.yahoo.com/v8/finance/chart/^VIX?range={range_param}&interval=1d", timeout=5)
                if vix_res.status_code == 200:
                    vix_data = vix_res.json()['chart']['result'][0]
                    vix_ts, vix_close = vix_data.get('timestamp', []), vix_data['indicators']['quote'][0].get('close')
                    if vix_ts and vix_close:
                        vix_df = pd.DataFrame({'VIX_Close': vix_close}, index=pd.to_datetime(vix_ts, unit='s')).dropna()
                        df = df.join(vix_df, how='left')
                        df['VIX_Close'] = df['VIX_Close'].ffill().bfill()
                    else: df['VIX_Close'] = 20.0
                else: df['VIX_Close'] = 20.0
            except: df['VIX_Close'] = 20.0
            df['VIX_MA_5'] = df['VIX_Close'].rolling(window=5).mean()
            
        return df, sirket_adi
    except: return None, None

def tek_hisse_analiz_et(h_kod):
    h_clean = str(h_kod).strip().upper()
    df_15m, sirket_adi = fetch_yahoo_data(h_clean, "15m", "5d")
    df_1d, _ = fetch_yahoo_data(h_clean, "1d", "3mo")

    if df_15m is None or df_1d is None or len(df_15m) < 30 or len(df_1d) < 30: return None

    son_fiyat = float(df_15m['Close'].iloc[-1])
    sma_20 = df_15m['Close'].rolling(window=20).mean()
    std_20 = df_15m['Close'].rolling(window=20).std()
    upper_band, lower_band = (sma_20 + (std_20 * 2)).iloc[-1], (sma_20 - (std_20 * 2)).iloc[-1]
    
    vol_sum = df_15m['Volume'].rolling(14).sum()
    vwap = (df_15m['Close'] * df_15m['Volume']).rolling(14).sum() / vol_sum
    son_vwap = round(vwap.iloc[-1], 2)
    son_rsi_15m = float(compute_rsi(df_15m['Close'], 14).iloc[-1])

    close_1d = df_1d['Close']
    atr_14_1d = float(compute_atr(df_1d, 14).iloc[-1])
    atr_yuzde = (atr_14_1d / son_fiyat) * 100
    son_rsi_1d = float(compute_rsi(close_1d, 14).iloc[-1])

    ema_getiri = close_1d.pct_change().dropna().ewm(span=10, adjust=False).mean().iloc[-1] * 100

    momentum = -0.15 if son_rsi_1d > 75 else (0.15 if son_rsi_1d < 30 else (son_rsi_1d - 50) / 100.0)
    vol_s_1d = df_1d['Volume'].tail(10)
    hacim_kati = float(vol_s_1d.iloc[-1] / vol_s_1d.mean()) if vol_s_1d.mean() > 0 else 1.0
    hacim_etkisi = min(hacim_kati, 2.0)

    tahminler_yuzde, kumbulatif_fiyat, aktif_momentum = {}, son_fiyat, momentum
    for gun in range(1, 4):
        gun_artis_yuzde = max(min(ema_getiri + (aktif_momentum * atr_yuzde * hacim_etkisi), atr_yuzde * 1.5), -atr_yuzde * 1.5)
        kumbulatif_fiyat *= (1 + (gun_artis_yuzde / 100))
        tahminler_yuzde[f'{gun}. Gün Tahmin (%)'] = round(((kumbulatif_fiyat - son_fiyat) / son_fiyat) * 100, 2)
        aktif_momentum *= 0.5
        hacim_etkisi = max(1.0, hacim_etkisi * 0.8)

    yarin_alis = round(son_fiyat - (atr_14_1d * 0.35), 2)
    potansiyel_pik = round(upper_band + (atr_14_1d * 0.1), 2)

    if son_fiyat >= upper_band and son_rsi_15m > 70:
        sinyal, strateji_metni, saatlik_yon = "PİK YAPTI (SAT)", "Fiyat Bollinger üst bandını aştı. Geri çekilme riski yüksek.", -1.0
    elif hacim_kati > 1.2 and son_fiyat > son_vwap and tahminler_yuzde['1. Gün Tahmin (%)'] > 0:
        sinyal, strateji_metni, saatlik_yon = "GÜÇLÜ AL & TUT", f"Fiyat VWAP üzerinde. Pik hedefi {potansiyel_pik} SEK.", 1.5
    elif son_fiyat < lower_band and son_rsi_15m < 30:
        sinyal, strateji_metni, saatlik_yon = "DİPTEN TEPKİ (AL)", f"Aşırı satıldı. {son_vwap} SEK seviyesine sıçrama beklenebilir.", 1.0
    else:
        sinyal, strateji_metni, saatlik_yon = "NÖTR (İZLE)", "Net bir trend yok, yatay seyir hakim.", 0.0

    beklenen_1h_getiri = round((atr_14_1d / son_fiyat / 4) * 100 * saatlik_yon, 2)
    # EKLENEN ÖZELLİK: 1 Saatlik Hedef Fiyat
    hedef_1h_fiyat = round(son_fiyat * (1 + (beklenen_1h_getiri / 100)), 2)

    return {
        'Şirket Adı': sirket_adi, 'Kod': h_clean, 'Son Fiyat (SEK)': round(son_fiyat, 2),
        '1 Saatlik Yön (%)': beklenen_1h_getiri,
        '1 Saatlik Hedef Fiyat (SEK)': hedef_1h_fiyat,  # Yeni Veri Eklendi!
        '1. Gün Tahmin (%)': tahminler_yuzde['1. Gün Tahmin (%)'], '2. Gün Tahmin (%)': tahminler_yuzde['2. Gün Tahmin (%)'], '3. Gün Tahmin (%)': tahminler_yuzde['3. Gün Tahmin (%)'],
        'Sinyal': sinyal, 'RSI (15m)': round(son_rsi_15m, 1), 'VWAP': son_vwap,
        'Potansiyel Pik': potansiyel_pik, 'Yarınki Alış': yarin_alis, 'Strateji': strateji_metni,
        'Analiz Zamanı': datetime.now().strftime("%Y-%m-%d %H:%M")
    }
