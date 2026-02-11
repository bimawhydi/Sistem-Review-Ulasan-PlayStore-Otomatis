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
# 2. PENGATURAN TEMA & CSS (NAVBAR HILANG + TOGGLE WARNA)
# ==========================================

# --- A. TOMBOL TOGGLE TEMA (DI SIDEBAR AGAR RAPI) ---
with st.sidebar:
    st.header("⚙️ Pengaturan")
    is_dark_mode = st.toggle("🌙 Mode Gelap (Dark Mode)", value=True)
    st.caption("Matikan untuk Mode Terang")

# --- B. LOGIKA WARNA ---
if is_dark_mode:
    bg_color = "#0E1117"
    text_color = "#FAFAFA"
    card_bg = "#262730"
    input_bg = "#353842"
    border_color = "#444444"
else:
    bg_color = "#FFFFFF"
    text_color = "#31333F"
    card_bg = "#F0F2F6"
    input_bg = "#FFFFFF"
    border_color = "#E0E0E0"

# --- C. INJEKSI CSS (SUPAYA WARNA KONSISTEN & NAVBAR HILANG) ---
st.markdown(f"""
<style>
    /* 1. HILANGKAN NAVBAR & FOOTER */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .block-container {{padding-top: 2rem;}}

    /* 2. PAKSA BACKGROUND UTAMA */
    .stApp {{
        background-color: {bg_color};
    }}

    /* 3. PAKSA WARNA TEKS */
    h1, h2, h3, h4, h5, h6, p, span, div, label, li {{
        color: {text_color} !important;
    }}

    /* 4. PERBAIKAN INPUT BOX */
    .stTextInput input, .stNumberInput input {{
        background-color: {input_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
    }}
    
    /* 5. STYLING KARTU/CONTAINER */
    div[data-testid="stExpander"], div[data-testid="stForm"] {{
        background-color: {card_bg};
        border-radius: 10px;
    }}
    
    /* 6. STYLING METRIC (KOTAK ANGKA) */
    div[data-testid="stMetric"] {{
        background-color: {card_bg};
        padding: 10px;
        border-radius: 8px;
        border: 1px solid {border_color};
    }}
    
    /* 7. TOMBOL PRIMARY */
    button[kind="primary"] {{
        background-color: #FF4B4B;
        color: white !important;
        border: none;
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD AI & FUNGSI
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

def buat_wordcloud(text_data, is_dark):
    # Background transparan biar aman di Dark/Light mode
    wc = WordCloud(
        width=800, height=400, 
        background_color=None, mode="RGBA",
        colormap='viridis' if not is_dark else 'Pastel1'
    ).generate(text_data)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    return plt

# ==========================================
# 4. TAMPILAN UTAMA
# ==========================================
st.title("🔮 Review Insight Pro")
st.markdown("Analisis sentimen ulasan Google Play Store otomatis dengan AI.")
st.write("---")

# INPUT (Menggunakan Container biar rapi)
with st.container():
    c1, c2 = st.columns([3, 1])
    with c1:
        input_url = st.text_input("🔗 Link Aplikasi:", placeholder="https://play.google.com/store/apps/details?id=com.gojek.app")
    with c2:
        jumlah_review = st.number_input("🔢 Jumlah:", min_value=10, max_value=2000, value=50, step=10)
    
    tombol = st.button("🚀 Analisa Sekarang", type="primary", use_container_width=True)

# ==========================================
# 5. LOGIKA & PERBAIKAN CHART
# ==========================================
if tombol:
    if not model:
        st.error("⚠️ File Model (.pkl) hilang.")
    elif not input_url:
        st.warning("⚠️ Masukkan link dulu.")
    else:
        app_id = ambil_app_id(input_url)
        if not app_id:
            st.error("❌ Link tidak valid.")
        else:
            try:
                with st.spinner('Sedang bekerja...'):
                    # SCRAPING
                    info_app = app(app_id, lang='id', country='id')
                    hasil_scrape, _ = reviews(
                        app_id, lang='id', country='id', sort=Sort.NEWEST, count=jumlah_review
                    )

                if len(hasil_scrape) == 0:
                    st.warning("Belum ada review.")
                else:
                    # PREDIKSI AI
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

                    # --- DASHBOARD ---
                    st.success("✅ Selesai!")
                    
                    # Info App
                    c_img, c_info = st.columns([1, 6])
                    with c_img:
                        st.image(info_app['icon'], width=80)
                    with c_info:
                        st.markdown(f"### {info_app['title']}")
                        st.caption(f"{info_app['developer']} | ⭐ {info_app['score']}")

                    st.markdown("---")

                    tab1, tab2, tab3 = st.tabs(["📊 Statistik", "☁️ WordCloud", "📝 Data"])

                    # TAB 1: GRAFIK (PERBAIKAN ERROR DI SINI)
                    with tab1:
                        tot = len(df)
                        pos = len(df[df['Sentimen'] == 'Positif'])
                        neg = len(df[df['Sentimen'] == 'Negatif'])
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total", tot)
                        m2.metric("Positif", pos, f"{pos/tot*100:.1f}%")
                        m3.metric("Negatif", neg, f"-{neg/tot*100:.1f}%", delta_color="inverse")
                        
                        st.write("#### Visualisasi")
                        g1, g2 = st.columns(2)
                        
                        with g1:
                            st.caption("Perbandingan")
                            base = alt.Chart(df).encode(theta=alt.Theta("count()", stack=True))
                            pie = base.mark_arc(innerRadius=60).encode(
                                color=alt.Color("Sentimen", scale=alt.Scale(domain=['Positif', 'Negatif'], range=['#2ECC71', '#E74C3C'])),
                                tooltip=["Sentimen", "count()"]
                            ).properties(height=300)
                            st.altair_chart(pie, use_container_width=True)
                            
                        with g2:
                            st.caption("Tren Harian")
                            # PERBAIKAN: Menamai kolom 'Jumlah' secara eksplisit
                            harian = df.groupby([pd.Grouper(key='Tanggal', freq='D'), 'Sentimen']).size().reset_index(name='Jumlah')
                            
                            # PERBAIKAN: Menambahkan ':Q' (Quantitative) agar Altair tahu ini angka
                            line = alt.Chart(harian).mark_line(point=True).encode(
                                x='Tanggal',
                                y='Jumlah:Q', 
                                color=alt.Color('Sentimen', scale=alt.Scale(domain=['Positif', 'Negatif'], range=['#2ECC71', '#E74C3C'])),
                                tooltip=['Tanggal', 'Sentimen', 'Jumlah']
                            ).properties(height=300)
                            st.altair_chart(line, use_container_width=True)

                    # TAB 2: WORDCLOUD
                    with tab2:
                        w1, w2 = st.columns(2)
                        with w1:
                            st.info("Positif")
                            tp = " ".join(df[df['Sentimen'] == 'Positif']['Review Bersih'])
                            if tp: st.pyplot(buat_wordcloud(tp, is_dark_mode))
                        with w2:
                            st.error("Negatif")
                            tn = " ".join(df[df['Sentimen'] == 'Negatif']['Review Bersih'])
                            if tn: st.pyplot(buat_wordcloud(tn, is_dark_mode))

                    # TAB 3: DATA
                    with tab3:
                        st.dataframe(df, use_container_width=True)
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download CSV", csv, "data.csv", "text/csv")

            except Exception as e:
                st.error(f"Terjadi Kesalahan: {e}")
