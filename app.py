import streamlit as st
import pandas as pd
from database import tek_hisse_analiz_et, db

st.set_page_config(page_title="XGBoost & ML Hisse Tahmin Portalı", layout="wide")

st.title("📈 Yapay Zeka (XGBoost) & Backtest Destekli Hisse Analizi")

# Hisse Seçim Alanı
hisseler = ["ELUX-B", "VOLV-B", "ERIC-B", "SEB-A", "SWED-A"]

col_input1, col_input2 = st.columns([3, 1])
with col_input1:
    selected_stock = st.selectbox(
        "Hisse Seçin:", 
        options=hisseler, 
        key="main_hisse_selectbox"  # Unique Key hatayı önler
    )
with col_input2:
    custom_stock = st.text_input("Veya Kod Girin (Örn: ERIC-B):", key="custom_hisse_text_input")

hisse_kodu = custom_stock.strip().upper() if custom_stock else selected_stock

if st.button("🚀 Yapay Zeka Analizini Başlat", key="btn_run_analysis"):
    with st.spinner(f"{hisse_kodu} için XGBoost modeli eğitiliyor ve backtest hesaplanıyor..."):
        res = tek_hisse_analiz_et(hisse_kodu)
        
        if res is None:
            st.error("Yeterli veri çekilemedi veya hisse koda ulaşılamadı. Lütfen sembolü kontrol edin.")
        else:
            st.success(f"Analiz Başarıyla Tamamlandı: {res['Şirket Adı']} ({res['Kod']})")
            
            # --- METRİKLER ---
            st.markdown("### 📊 Anlık Durum ve Yapay Zeka Metrikleri")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Son Fiyat", f"{res['Son Fiyat (SEK)']} SEK")
            m2.metric(
                "1 Saatlik Hedef Fiyat", 
                f"{res['1 Saatlik Hedef Fiyat (SEK)']} SEK", 
                delta=f"%{res['1 Saatlik Yön (%)']}"
            )
            m3.metric("ML Backtest Başarısı", f"%{res['ML Kazanma Oranı (%)']}")
            m4.metric("RSI (15m)", res['RSI (15m)'])
            
            st.divider()

            # --- SİNYAL VE STRATEJİ ---
            st.markdown("### 🤖 ML Sinyal ve Strateji Önerisi")
            if "AL" in res['Sinyal']:
                st.success(f"**Sinyal:** {res['Sinyal']}\n\n**Açıklama:** {res['Strateji']}")
            elif "SAT" in res['Sinyal']:
                st.error(f"**Sinyal:** {res['Sinyal']}\n\n**Açıklama:** {res['Strateji']}")
            else:
                st.info(f"**Sinyal:** {res['Sinyal']}\n\n**Açıklama:** {res['Strateji']}")

            st.divider()

            # --- TAHMİN TABLE VE TEKNİK SEVİYELER ---
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("### 📅 Çoklu Gün Tahmin Projeksiyonu")
                df_tahmin = pd.DataFrame({
                    "Tahmin Periyodu": ["1. Gün", "2. Gün", "3. Gün"],
                    "Beklenen Değişim (%)": [
                        f"%{res['1. Gün Tahmin (%)']}",
                        f"%{res['2. Gün Tahmin (%)']}",
                        f"%{res['3. Gün Tahmin (%)']}"
                    ]
                })
                st.table(df_tahmin)
            
            with col_right:
                st.markdown("### 🎯 Kritik Teknik Seviyeler")
                st.write(f"**VWAP (Denge Fiyatı):** {res['VWAP']} SEK")
                st.write(f"**Potansiyel Pik (Direnç):** {res['Potansiyel Pik']} SEK")
                st.write(f"**Yarınki İdeal Alış:** {res['Yarınki Alış']} SEK")
                st.caption(f"Analiz Zamanı: {res['Analiz Zamanı']}")
