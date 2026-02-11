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
# 2. HEADER & TOGGLE TEMA
# ==========================================
col_title, col_toggle = st.columns([8, 2])

with col_title:
    st.markdown("## 🔮 Review Insight Pro")
    st.caption("Analisis sentimen ulasan Google Play Store otomatis dengan AI.")

with col_toggle:
    # Toggle Switch
    is_dark_mode = st.toggle("🌙 Mode Gelap", value=False)

# ==========================================
# 3. LOGIKA CSS (STYLING)
# ==========================================
if is_dark_mode:
    # --- DARK MODE ---
    bg_color = "#0E1117"
    text_color = "#FAFAFA"
    card_bg = "#262730"
    input_bg = "#353842"
    border_color = "#444444"
    placeholder_color = "#AAAAAA" # Warna teks samar
else:
    # --- LIGHT MODE ---
    bg_color = "#FFFFFF"
    text_color = "#31333F"
    card_bg = "#F9F9F9"
    input_bg = "#FFFFFF"
    border_color = "#E0E0E0"
    placeholder_color = "#666666"

# INJEKSI CSS YANG LEBIH KUAT
st.markdown(f"""
<style>
    /* 1. Background Utama */
    .stApp {{
        background-color: {bg_color};
    }}

    /* 2. Warna Teks Global */
    h1, h2, h3, h4, h5, h6, p, li, span, div {{
        color: {text_color} !important;
    }}

    /* 3. Perbaikan Input Box (Biar Teks Terbaca) */
    .stTextInput input, .stNumberInput input {{
        background-color: {input_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
    }}
    
    /* 4. Perbaikan Warna Placeholder (Teks Samar) */
    ::placeholder {{
        color: {placeholder_color} !important;
        opacity: 1;
    }}

    /* 5. Perbaikan Label di atas Input */
    .stTextInput label, .stNumberInput label {{
        color: {text_color} !important;
    }}

    /* 6. Styling Kartu Statistik (Metric) */
    div[data-testid="stMetric"] {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        padding: 10px;
        border-radius: 8px;
        color: {text_color};
    }}

    /* 7. Styling Expander/Container */
    div[data-testid="stExpander"], div.css-1r6slb0 {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        border-radius: 8px;
    }}

    /* 8. Tombol Analisa (Merah) */
    button[kind="primary"] {{
        background-color: #FF4B4B;
        color: white !important;
        border: none;
        font-weight: bold;
    }}
    button[kind="primary"]:hover {{
        background-color: #D93030;
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. LOAD AI
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

# Fungsi helper lainnya tetap sama
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
    bg = 'black' if is_dark else 'white'
    cmap = 'Pastel1' if is_dark else 'viridis'
    wc = WordCloud(width=800, height=400, background_color=bg, colormap=cmap).generate(text_data)
    plt.figure(figsize=(10, 5), facecolor=bg)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    return plt

# ==========================================
# 5. INPUT SECTION
# ==========================================
st.markdown("---")

# Container Utama
with st.container():
    c1, c2 = st.columns([3, 1])
    
    with c1:
        input_url = st.text_input(
            "🔗 Tempel Link Aplikasi Play Store:", 
            placeholder="https://play.google.com/store/apps/details?id=com.mobile.legends"
        )
    
    with c2:
        jumlah_review = st.number_input("🔢 Jumlah Data:", min_value=10, max_value=2000, value=50, step=10)
    
    tombol = st.button("🚀 Analisa Sekarang", type="primary", use_container_width=True)

# ==========================================
# 6. OUTPUT SECTION
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
                with st.spinner('Sedang bekerja...'):
                    info_app = app(app_id, lang='id', country='id')
                    hasil_scrape, _ = reviews(
                        app_id, lang='id', country='id', sort=Sort.NEWEST, count=jumlah_review
                    )

                if len(hasil_scrape) == 0:
                    st.warning("Belum ada review.")
                else:
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

                    # HASIL
                    st.success("✅ Selesai!")
                    
                    # Header Info App
                    with st.container():
                        st.image(info_app['icon'], width=80)
                        st.markdown(f"### {info_app['title']}")
                        st.caption(f"{info_app['developer']} | ⭐ {info_app['score']}")

                    st.markdown("---")

                    # TAB
                    t1, t2, t3 = st.tabs(["📊 Statistik", "☁️ WordCloud", "📝 Data"])

                    with t1:
                        tot = len(df)
                        pos = len(df[df['Sentimen'] == 'Positif'])
                        neg = len(df[df['Sentimen'] == 'Negatif'])
                        
                        k1, k2, k3 = st.columns(3)
                        k1.metric("Total Review", tot)
                        k2.metric("Positif", pos, f"{pos/tot*100:.1f}%")
                        k3.metric("Negatif", neg, f"-{neg/tot*100:.1f}%", delta_color="inverse")
                        
                        st.write("####")
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
                            harian = df.groupby([pd.Grouper(key='Tanggal', freq='D'), 'Sentimen']).size().reset_index(name='Jml')
                            line = alt.Chart(harian).mark_line(point=True).encode(
                                x='Tanggal',
                                y='Jml',
                                color=alt.Color('Sentimen', scale=alt.Scale(domain=['Positif', 'Negatif'], range=['#2ECC71', '#E74C3C']))
                            ).properties(height=300)
                            st.altair_chart(line, use_container_width=True)

                    with t2:
                        w1, w2 = st.columns(2)
                        with w1:
                            st.info("Positif")
                            tp = " ".join(df[df['Sentimen'] == 'Positif']['Review Bersih'])
                            if tp: st.pyplot(buat_wordcloud(tp, is_dark_mode))
                        with w2:
                            st.error("Negatif")
                            tn = " ".join(df[df['Sentimen'] == 'Negatif']['Review Bersih'])
                            if tn: st.pyplot(buat_wordcloud(tn, is_dark_mode))

                    with t3:
                        st.dataframe(df, use_container_width=True)
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download CSV", csv, "data.csv", "text/csv")

            except Exception as e:
                st.error(f"Error: {e}")
