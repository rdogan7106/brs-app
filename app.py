import streamlit as st
import pandas as pd
from database import tek_hisse_analiz_et, db

st.set_page_config(page_title="Hibrit Hisse & ML Portalı", layout="wide")

st.title("📈 BIST / İsveç Borsası Hibrit Analiz & Yapay Zeka Portalı")

# Tüm özellikleri sekmeler altında topluyoruz
tab1, tab2, tab3 = st.tabs([
    "🤖 Yapay Zeka (XGBoost) Tek Hisse", 
    "📊 Toplu Hisse Taraması", 
    "📜 Veritabanı & Geçmiş Kayıtlar"
])

hisseler_varsayilan = ["ELUX-B", "VOLV-B", "ERIC-B", "SEB-A", "SWED-A"]

# ==========================================
# 1. SEKME: TEK HİSSE & ML ANALİZİ
# ==========================================
with tab1:
    st.subheader("Tek Hisse İçin Detaylı ML ve Backtest Analizi")
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_stock = st.selectbox("Hisse Seçin:", options=hisseler_varsayilan, key="tab1_select")
    with col2:
        custom_stock = st.text_input("Veya Kod Girin (Örn: ERIC-B):", key="tab1_input")

    hisse_kodu = custom_stock.strip().upper() if custom_stock else selected_stock

    if st.button("🚀 Tek Hisse Analiz Et", key="btn_tab1"):
        with st.spinner(f"{hisse_kodu} analiz ediliyor..."):
            res = tek_hisse_analiz_et(hisse_kodu)
            if res is None:
                st.error("Veri çekilemedi, lütfen sembolü kontrol edin.")
            else:
                st.success(f"Analiz Tamamlandı: {res['Şirket Adı']} ({res['Kod']})")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Son Fiyat", f"{res['Son Fiyat (SEK)']} SEK")
                m2.metric("1 Saatlik Hedef Fiyat", f"{res['1 Saatlik Hedef Fiyat (SEK)']} SEK", delta=f"%{res['1 Saatlik Yön (%)']}")
                m3.metric("ML Backtest Başarısı", f"%{res['ML Kazanma Oranı (%)']}")
                m4.metric("RSI (15m)", res['RSI (15m)'])
                
                st.divider()

                if "AL" in res['Sinyal']:
                    st.success(f"**Sinyal:** {res['Sinyal']} | **Strateji:** {res['Strateji']}")
                elif "SAT" in res['Sinyal']:
                    st.error(f"**Sinyal:** {res['Sinyal']} | **Strateji:** {res['Strateji']}")
                else:
                    st.info(f"**Sinyal:** {res['Sinyal']} | **Strateji:** {res['Strateji']}")

                st.divider()

                col_l, col_r = st.columns(2)
                with col_l:
                    st.markdown("### 📅 3 Günlük Tahmin Projeksiyonu")
                    df_tahmin = pd.DataFrame({
                        "Tahmin Periyodu": ["1. Gün", "2. Gün", "3. Gün"],
                        "Beklenen Değişim": [
                            f"%{res['1. Gün Tahmin (%)']}",
                            f"%{res['2. Gün Tahmin (%)']}",
                            f"%{res['3. Gün Tahmin (%)']}"
                        ]
                    })
                    st.table(df_tahmin)
                with col_r:
                    st.markdown("### 🎯 Teknik Seviyeler")
                    st.write(f"**VWAP:** {res['VWAP']} SEK")
                    st.write(f"**Potansiyel Pik:** {res['Potansiyel Pik']} SEK")
                    st.write(f"**Yarınki İdeal Alış:** {res['Yarınki Alış']} SEK")
                    st.caption(f"Analiz Zamanı: {res['Analiz Zamanı']}")

# ==========================================
# 2. SEKME: TOPLU HİSSE TARAMASI
# ==========================================
with tab2:
    st.subheader("Tüm İzleme Listesini Anlık Tarama")
    st.write("Aşağıdaki buton listelenen tüm hisseleri tek tıkla analiz eder ve tablo halinde sunar.")
    
    if st.button("🔄 Tüm Listeyi Tara ve Karşılaştır", key="btn_tab2"):
        results = []
        bar = st.progress(0)
        for idx, kod in enumerate(hisseler_varsayilan):
            res = tek_hisse_analiz_et(kod)
            if res:
                results.append(res)
            bar.progress((idx + 1) / len(hisseler_varsayilan))
            
        if results:
            df_all = pd.DataFrame(results)
            st.dataframe(df_all, use_container_width=True)
        else:
            st.warning("Veri çekilemedi.")

# ==========================================
# 3. SEKME: VERİTABANI & GEÇMİŞ
# ==========================================
with tab3:
    st.subheader("Sistem Veritabanı ve Geçmiş Durumu")
    st.info("SQLite veritabanı bağlandı (`app_data.db`). Tüm geçmiş verileriniz burada saklanır.")
