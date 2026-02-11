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
# 2. HEADER & THEME TOGGLE (NO SIDEBAR)
# ==========================================
# Kita buat layout 2 kolom: Kiri (Judul), Kanan (Tombol Toggle)
col_header, col_toggle = st.columns([6, 1])

with col_header:
    st.markdown("## 🔮 Review Insight Pro")

with col_toggle:
    st.write("") # Spasi kosong biar sejajar
    # Tombol Toggle di pojok kanan atas
    is_dark_mode = st.toggle("🌙 Dark Mode", value=False)

# ==========================================
# 3. LOGIKA WARNA (CSS INJECTION)
# ==========================================
if is_dark_mode:
    # --- PALET DARK MODE ---
    bg_color = "#0E1117"        # Background Utama Gelap
    text_color = "#FAFAFA"      # Teks Putih
    card_bg = "#262730"         # Background Kartu/Container
    input_bg = "#353842"        # Background Kotak Input
    metric_bg = "#1F2937"       # Background Angka Statistik
    border_color = "#444444"
else:
    # --- PALET LIGHT MODE ---
    bg_color = "#FFFFFF"        # Background Utama Putih
    text_color = "#31333F"      # Teks Hitam Abu
    card_bg = "#F8F9FA"         # Background Kartu
    input_bg = "#FFFFFF"        # Background Kotak Input
    metric_bg = "#FFFFFF"       # Background Angka Statistik
    border_color = "#E0E0E0"

# CSS SAKTI UNTUK MEMAKSA WARNA
st.markdown(f"""
<style>
    /* 1. Background Utama */
    .stApp {{
        background-color: {bg_color};
    }}

    /* 2. Mengubah SEMUA Teks (Judul, Paragraf, Label Input) */
    h1, h2, h3, h4, h5, h6, p, span, div, label {{
        color: {text_color} !important;
    }}

    /* 3. Khusus Label di atas Input Box (Sering bermasalah) */
    .stTextInput label, .stNumberInput label {{
        color: {text_color} !important;
    }}
    
    /* 4. Mengubah Warna Kotak Input (Tempat ngetik) */
    .stTextInput > div > div, .stNumberInput > div > div {{
        background-color: {input_bg};
        color: {text_color};
        border-color: {border_color};
    }}
    
    /* 5. Warna Teks yang diketik user di dalam kotak */
    input {{
        color: {text_color} !important;
    }}

    /* 6. Styling Container/Card (Expander & Kotak) */
    div[data-testid="stExpander"], div.css-1r6slb0 {{
        background-color: {card_bg};
        border-radius: 10px;
        border: 1px solid {border_color};
    }}
    
    /* 7. Styling Metric (Kotak Statistik) */
    div[data-testid="stMetric"] {{
        background-color: {metric_bg};
        border: 1px solid {border_color};
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }}
    
    /* 8. Tombol Primary (Merah) */
    button[kind="primary"] {{
        background-color: #FF4B4B;
        color: white !important; /* Teks tombol selalu putih */
        border: none;
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. LOAD AI & FUNGSI BANTUAN
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
    bg = 'black' if is_dark else 'white'
    cmap = 'Pastel1' if is_dark else 'viridis'
    wc = WordCloud(width=800, height=400, background_color=bg, colormap=cmap).generate(text_data)
    plt.figure(figsize=(10, 5), facecolor=bg)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    return plt

# ==========================================
# 5. INPUT SECTION (MAIN AREA)
# ==========================================
st.write("Analisis sentimen ulasan Google Play Store otomatis dengan AI.")

# Container Input dengan Border Halus
with st.container(border=True):
    col_input, col_num = st.columns([4, 1])
    
    with col_input:
        input_url = st.text_input("🔗 Tempel Link Aplikasi Play Store:", placeholder="Contoh: https://play.google.com/store/apps/details?id=com.gojek.app")
    
    with col_num:
        jumlah_review = st.number_input("🔢 Jumlah Data:", min_value=10, max_value=2000, value=50, step=10)
    
    tombol_analisa = st.button("🚀 Analisa Sekarang", type="primary", use_container_width=True)

# ==========================================
# 6. LOGIKA & TAMPILAN HASIL
# ==========================================
if tombol_analisa:
    if not model:
        st.error("⚠️ File Model (.pkl) tidak ditemukan.")
    elif not input_url:
        st.warning("⚠️ Masukkan link aplikasi dulu ya!")
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
                    st.warning("Belum ada review untuk aplikasi ini.")
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

                    # --- HASIL DASHBOARD ---
                    st.success("✅ Selesai!")
                    
                    # Info Aplikasi
                    c1, c2 = st.columns([1, 6])
                    with c1:
                        st.image(info_app['icon'], width=90)
                    with c2:
                        st.markdown(f"### {info_app['title']}")
                        st.caption(f"{info_app['developer']} | ⭐ {info_app['score']}")

                    st.markdown("---")

                    # TAB MENU
                    tab1, tab2, tab3 = st.tabs(["📊 Statistik", "☁️ WordCloud", "📝 Data Tabel"])

                    # TAB 1: STATISTIK
                    with tab1:
                        total = len(df)
                        pos = len(df[df['Sentimen'] == 'Positif'])
                        neg = len(df[df['Sentimen'] == 'Negatif'])

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total Review", total)
                        m2.metric("Positif", pos, f"{pos/total*100:.1f}%")
                        m3.metric("Negatif", neg, f"-{neg/total*100:.1f}%", delta_color="inverse")

                        st.markdown("###")
                        
                        g1, g2 = st.columns(2)
                        with g1:
                            st.write("**Komposisi Sentimen**")
                            base = alt.Chart(df).encode(theta=alt.Theta("count()", stack=True))
                            pie = base.mark_arc(innerRadius=60).encode(
                                color=alt.Color("Sentimen", scale=alt.Scale(domain=['Positif', 'Negatif'], range=['#2ECC71', '#E74C3C'])),
                                tooltip=["Sentimen", "count()"]
                            ).properties(height=300)
                            st.altair_chart(pie, use_container_width=True)

                        with g2:
                            st.write("**Tren Sentimen**")
                            daily = df.groupby([pd.Grouper(key='Tanggal', freq='D'), 'Sentimen']).size().reset_index(name='Jumlah')
                            line = alt.Chart(daily).mark_line(point=True).encode(
                                x='Tanggal',
                                y='Jumlah',
                                color=alt.Color('Sentimen', scale=alt.Scale(domain=['Positif', 'Negatif'], range=['#2ECC71', '#E74C3C'])),
                                tooltip=['Tanggal', 'Sentimen', 'Jumlah']
                            ).properties(height=300)
                            st.altair_chart(line, use_container_width=True)

                    # TAB 2: WORDCLOUD
                    with tab2:
                        wc1, wc2 = st.columns(2)
                        with wc1:
                            st.success("Kata Kunci Positif")
                            txt_pos = " ".join(df[df['Sentimen'] == 'Positif']['Review Bersih'])
                            if txt_pos: st.pyplot(buat_wordcloud(txt_pos, is_dark_mode))
                        with wc2:
                            st.error("Kata Kunci Negatif")
                            txt_neg = " ".join(df[df['Sentimen'] == 'Negatif']['Review Bersih'])
                            if txt_neg: st.pyplot(buat_wordcloud(txt_neg, is_dark_mode))

                    # TAB 3: DATA RAW
                    with tab3:
                        st.dataframe(df, use_container_width=True)
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download CSV", csv, f"review_{app_id}.csv", "text/csv")

            except Exception as e:
                st.error(f"Terjadi Kesalahan: {e}")