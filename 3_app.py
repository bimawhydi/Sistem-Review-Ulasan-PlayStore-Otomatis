import streamlit as st
import pandas as pd
import joblib
import re
import string
import altair as alt
from google_play_scraper import reviews, Sort, app
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Review Insight Pro",
    page_icon="🔮",
    layout="wide"
)

# ==========================================
# 2. CSS PEMBERSIH (HANYA UNTUK HIDE NAVBAR)
# ==========================================
# Kita tidak lagi mengubah warna teks secara paksa, 
# biar Streamlit yang mengatur kontrasnya agar selalu terbaca.
st.markdown("""
<style>
    /* Sembunyikan Menu, Footer, Header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Naikkan konten ke atas */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }
    
    /* Styling Tombol agar lebih 'pop' */
    button[kind="primary"] {
        background-color: #FF4B4B;
        border: none;
        transition: 0.3s;
    }
    button[kind="primary"]:hover {
        background-color: #FF0000;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD AI & FUNGSI BANTUAN
# ==========================================
@st.cache_resource
def load_ai():
    try:
        model = joblib.load('model_svm_cerdas.pkl')
        vectorizer = joblib.load('vectorizer_kamus.pkl')
        return model, vectorizer
    except:
        return None, None

model, vectorizer = load_ai()

def bersihkan_teks(teks):
    teks = teks.lower()
    teks = re.sub(r"\d+", "", teks)
    teks = teks.translate(str.maketrans("", "", string.punctuation))
    teks = teks.strip()
    return teks

def ambil_app_id(url):
    match = re.search(r'id=([a-zA-Z0-9\._]+)', url)
    return match.group(1) if match else None

# FUNGSI WORDCLOUD TRANSPARAN (FIX TEMA)
def buat_wordcloud(text_data):
    # background_color=None dan mode="RGBA" membuat background transparan
    # jadi aman untuk Dark Mode maupun Light Mode
    wc = WordCloud(
        width=800, 
        height=400, 
        background_color=None, 
        mode="RGBA", 
        colormap='viridis',
        regexp=r"\w[\w']+"
    ).generate(text_data)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    return plt

# ==========================================
# 4. HEADER UI
# ==========================================
st.title("🔮 Review Insight Pro")
st.markdown("Analisis sentimen ulasan Google Play Store otomatis dengan AI.")

st.write("---")

# ==========================================
# 5. INPUT SECTION (CARD STYLE)
# ==========================================
# Menggunakan st.container(border=True) membuat kotak cantik otomatis
# yang warnanya menyesuaikan tema (Putih/Gelap) tanpa bug.
with st.container(border=True):
    st.subheader("🔍 Mulai Analisis")
    
    col_input, col_num = st.columns([3, 1])
    
    with col_input:
        input_url = st.text_input(
            "Link Google Play Store:", 
            placeholder="https://play.google.com/store/apps/details?id=com.mobile.legends",
            help="Copy link aplikasi dari browser dan paste di sini"
        )
    
    with col_num:
        jumlah_review = st.number_input("Jumlah Data:", min_value=10, max_value=2000, value=50, step=10)
    
    tombol = st.button("🚀 Analisa Sekarang", type="primary", use_container_width=True)

# ==========================================
# 6. LOGIKA UTAMA
# ==========================================
if tombol:
    if not model:
        st.error("⚠️ File Model (.pkl) tidak ditemukan.")
    elif not input_url:
        st.warning("⚠️ Masukkan link aplikasi dulu.")
    else:
        app_id = ambil_app_id(input_url)
        if not app_id:
            st.error("❌ Link tidak valid.")
        else:
            try:
                with st.spinner('Sedang mengambil data & berpikir...'):
                    # SCRAPING
                    info_app = app(app_id, lang='id', country='id')
                    hasil_scrape, _ = reviews(
                        app_id, lang='id', country='id', sort=Sort.NEWEST, count=jumlah_review
                    )

                if len(hasil_scrape) == 0:
                    st.warning("Belum ada review untuk aplikasi ini.")
                else:
                    # AI PREDICTION
                    data_hasil = []
                    for item in hasil_scrape:
                        clean = bersihkan_teks(item['content'])
                        sentimen = model.predict(vectorizer.transform([clean]))[0]
                        data_hasil.append({
                            'User': item['userName'],
                            'Review': item['content'],
                            'Review Bersih': clean,
                            'Sentimen': sentimen,
                            'Tanggal': pd.to_datetime(item['at'])
                        })
                    
                    df = pd.DataFrame(data_hasil)

                    # --- HASIL DASHBOARD ---
                    st.success("✅ Analisis Selesai!")
                    
                    # Info Aplikasi
                    with st.container(border=True):
                        c1, c2 = st.columns([1, 6])
                        with c1:
                            st.image(info_app['icon'], width=100)
                        with c2:
                            st.markdown(f"### {info_app['title']}")
                            st.caption(f"Developer: {info_app['developer']} | Genre: {info_app['genre']}")
                            st.write(f"Rating: ⭐ **{info_app['score']}**")

                    # TAB MENU
                    tab1, tab2, tab3 = st.tabs(["📊 Statistik", "☁️ WordCloud", "📝 Data Tabel"])

                    # TAB 1: STATISTIK
                    with tab1:
                        # Metric Cards
                        tot = len(df)
                        pos = len(df[df['Sentimen'] == 'Positif'])
                        neg = len(df[df['Sentimen'] == 'Negatif'])
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total Review", tot)
                        m2.metric("Positif", pos, f"{pos/tot*100:.1f}%")
                        m3.metric("Negatif", neg, f"-{neg/tot*100:.1f}%", delta_color="inverse")
                        
                        st.write("#### Tren & Visualisasi")
                        g1, g2 = st.columns(2)
                        
                        with g1:
                            st.caption("Perbandingan Sentimen")
                            base = alt.Chart(df).encode(theta=alt.Theta("count()", stack=True))
                            pie = base.mark_arc(innerRadius=60).encode(
                                color=alt.Color("Sentimen", scale=alt.Scale(domain=['Positif', 'Negatif'], range=['#2ECC71', '#E74C3C'])),
                                tooltip=["Sentimen", "count()"]
                            ).properties(height=300)
                            st.altair_chart(pie, use_container_width=True)
                            
                        with g2:
                            st.caption("Tren Harian")
                            harian = df.groupby([pd.Grouper(key='Tanggal', freq='D'), 'Sentimen']).size().reset_index(name='Jml')
                            line = alt.Chart(harian).mark_line(point=True).encode(
                                x='Tanggal',
                                y='Jml',
                                color=alt.Color('Sentimen', scale=alt.Scale(domain=['Positif', 'Negatif'], range=['#2ECC71', '#E74C3C'])),
                                tooltip=['Tanggal', 'Sentimen', 'Jumlah']
                            ).properties(height=300)
                            st.altair_chart(line, use_container_width=True)

                    # TAB 2: WORD CLOUD
                    with tab2:
                        w1, w2 = st.columns(2)
                        with w1:
                            st.success("Topik Positif")
                            txt_pos = " ".join(df[df['Sentimen'] == 'Positif']['Review Bersih'])
                            if txt_pos: st.pyplot(buat_wordcloud(txt_pos))
                            else: st.info("Tidak ada data positif")
                                
                        with w2:
                            st.error("Topik Negatif")
                            txt_neg = " ".join(df[df['Sentimen'] == 'Negatif']['Review Bersih'])
                            if txt_neg: st.pyplot(buat_wordcloud(txt_neg))
                            else: st.info("Tidak ada data negatif")

                    # TAB 3: DATA RAW
                    with tab3:
                        # Dataframe native Streamlit (Paling aman & rapi)
                        st.dataframe(
                            df[['Tanggal', 'User', 'Rating', 'Review', 'Sentimen']], 
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"Review_{info_app['title']}.csv",
                            mime="text/csv",
                            type="primary"
                        )

            except Exception as e:
                st.error(f"Terjadi Kesalahan: {e}")
