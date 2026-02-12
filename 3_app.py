import streamlit as st
import pandas as pd
import joblib
import re
import string
import altair as alt
from google_play_scraper import reviews, Sort, app
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# =========================================================
# 1) PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Review Insight Pro",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",  # sidebar disembunyikan by default
)

# =========================================================
# 2) DARK THEME TOKENS (ONLY)
# =========================================================
T = {
    "bg": "#0B1020",
    "panel": "#121A2E",
    "card": "#121A2E",
    "card2": "#0F172A",
    "text": "#EAF0FF",
    "muted": "#A7B0C0",
    "border": "rgba(255,255,255,.08)",
    "primary": "#4F8CFF",
    "primary2": "#22D3EE",
    "danger": "#FF4B4B",
    "success": "#22C55E",
    "warning": "#FBBF24",
    "input": "#0B1226",
}

# =========================================================
# 3) GLOBAL CSS (MODERN DARK UI)
# =========================================================
st.markdown(
    f"""
<style>
    /* Hide Streamlit chrome */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .block-container {{padding-top: 1.1rem; padding-bottom: 2rem;}}

    /* (Optional) sembunyikan sidebar sepenuhnya */
    section[data-testid="stSidebar"] {{display: none;}}

    .stApp {{
        background: radial-gradient(1200px 600px at 15% 5%, rgba(79,140,255,.22), transparent 55%),
                    radial-gradient(900px 500px at 90% 10%, rgba(34,211,238,.18), transparent 50%),
                    {T["bg"]};
        color: {T["text"]};
    }}

    h1, h2, h3, h4, h5, h6, p, span, div, label, li {{
        color: {T["text"]} !important;
    }}
    .muted {{ color: {T["muted"]} !important; }}

    .card {{
        background: linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,0)) , {T["card"]};
        border: 1px solid {T["border"]};
        border-radius: 16px;
        padding: 16px 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,.10);
    }}
    .card-tight {{
        padding: 12px 14px;
        border-radius: 14px;
    }}

    .hero {{
        border-radius: 18px;
        padding: 18px 18px;
        border: 1px solid {T["border"]};
        background:
          radial-gradient(700px 300px at 20% 0%, rgba(79,140,255,.25), transparent 60%),
          radial-gradient(600px 260px at 85% 10%, rgba(34,211,238,.22), transparent 60%),
          {T["panel"]};
        box-shadow: 0 14px 40px rgba(0,0,0,.12);
    }}
    .hero-title {{
        font-size: 30px;
        font-weight: 800;
        margin: 0;
        letter-spacing: .2px;
    }}
    .hero-sub {{
        margin-top: 6px;
        color: {T["muted"]};
        font-size: 14px;
    }}
    .chip {{
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid {T["border"]};
        background: rgba(79,140,255,.12);
        color: {T["text"]};
        font-size: 12px;
        margin-right: 8px;
    }}

    .stTextInput input, .stNumberInput input {{
        background: {T["input"]} !important;
        border: 1px solid {T["border"]} !important;
        border-radius: 12px !important;
        padding: 10px 12px !important;
        color: {T["text"]} !important;
    }}

    button[kind="primary"] {{
        background: linear-gradient(90deg, {T["primary"]}, {T["primary2"]}) !important;
        border: none !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 0.6rem 0.9rem !important;
        box-shadow: 0 10px 22px rgba(37, 99, 235, .25);
    }}
    button[kind="secondary"] {{
        border-radius: 12px !important;
        border: 1px solid {T["border"]} !important;
        color: {T["text"]} !important;
    }}

    div[data-testid="stMetric"] {{
        background: {T["card2"]};
        border: 1px solid {T["border"]};
        border-radius: 16px;
        padding: 12px 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,.08);
    }}

    button[data-baseweb="tab"] {{
        background: transparent !important;
        border-radius: 12px !important;
        padding: 10px 12px !important;
        border: 1px solid transparent !important;
        color: {T["muted"]} !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        border: 1px solid {T["border"]} !important;
        background: rgba(79,140,255,.10) !important;
        color: {T["text"]} !important;
    }}

    .stDataFrame {{
        border: 1px solid {T["border"]};
        border-radius: 14px;
        overflow: hidden;
    }}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# 4) LOAD AI + HELPERS
# =========================================================
@st.cache_resource
def load_ai():
    try:
        model = joblib.load("model_svm_cerdas.pkl")
        vectorizer = joblib.load("vectorizer_kamus.pkl")
        return model, vectorizer
    except:
        return None, None

model, vectorizer = load_ai()

def bersihkan_teks(teks: str) -> str:
    teks = teks.lower()
    teks = re.sub(r"\d+", "", teks)
    teks = teks.translate(str.maketrans("", "", string.punctuation))
    return teks.strip()

def ambil_app_id(url: str):
    match = re.search(r"id=([a-zA-Z0-9\._]+)", url)
    return match.group(1) if match else None

def buat_wordcloud(text_data: str):
    wc = WordCloud(
        width=900,
        height=450,
        background_color=None,
        mode="RGBA",
        colormap="Pastel1",  # dark friendly
    ).generate(text_data)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    return plt

# =========================================================
# 5) SESSION STATE (results)
# =========================================================
if "df" not in st.session_state:
    st.session_state.df = None
if "info_app" not in st.session_state:
    st.session_state.info_app = None

# =========================================================
# 6) HERO HEADER
# =========================================================
st.markdown(
    """
<div class="hero">
  <div class="chip">AI Sentiment</div>
  <div class="chip">Google Play Scraper</div>
  <div class="chip">Dashboard</div>
  <p class="hero-title">Review Insight Pro</p>
  <p class="hero-sub">Analisis sentimen ulasan Google Play Store otomatis dengan AI — tampilan dark modern, rapi, dan nyaman dilihat.</p>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

# =========================================================
# NAV: TABS (Analisis / Dashboard / Data)
# =========================================================
tab_analisis, tab_dashboard, tab_data = st.tabs(["📌 Analisis", "📊 Dashboard", "📝 Data"])

# =========================================================
# TAB ANALISIS
# =========================================================
with tab_analisis:
    left, right = st.columns([1.35, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🚀 Mulai Analisis")
        st.caption("Masukkan link aplikasi Google Play dan jumlah review yang ingin dianalisis.")

        input_url = st.text_input(
            "🔗 Link Aplikasi",
            placeholder="https://play.google.com/store/apps/details?id=com.gojek.app",
            key="input_url",
        )
        jumlah_review = st.number_input(
            "🔢 Jumlah Review",
            min_value=10,
            max_value=2000,
            value=50,
            step=10,
            key="jumlah_review",
        )

        colA, colB = st.columns([1, 1])
        with colA:
            run = st.button("Analisa Sekarang", type="primary", use_container_width=True, key="run_btn")
        with colB:
            clear = st.button("Reset Hasil", type="secondary", use_container_width=True, key="clear_btn")

        if clear:
            st.session_state.df = None
            st.session_state.info_app = None
            st.toast("Hasil di-reset.", icon="🧹")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ℹ️ Tips")
        st.markdown(
            """
- Pastikan link **mengandung** `id=com.nama.app`
- Jika model `.pkl` tidak terbaca, cek nama file dan lokasi.
- Untuk insight lebih lengkap, naikkan jumlah review (misal 200–500).
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if run:
        if not model:
            st.error("⚠️ File Model (.pkl) hilang / tidak terbaca.")
        elif not input_url:
            st.warning("⚠️ Masukkan link dulu.")
        else:
            app_id = ambil_app_id(input_url)
            if not app_id:
                st.error("❌ Link tidak valid (app id tidak ditemukan).")
            else:
                try:
                    with st.spinner("Sedang bekerja..."):
                        info_app = app(app_id, lang="id", country="id")
                        hasil_scrape, _ = reviews(app_id, lang="id", country="id", sort=Sort.NEWEST, count=jumlah_review)

                    if len(hasil_scrape) == 0:
                        st.warning("Belum ada review.")
                    else:
                        data_hasil = []
                        for item in hasil_scrape:
                            clean = bersihkan_teks(item["content"])
                            sentimen = model.predict(vectorizer.transform([clean]))[0]
                            data_hasil.append(
                                {
                                    "User": item["userName"],
                                    "Review": item["content"],
                                    "Review Bersih": clean,
                                    "Sentimen": sentimen,
                                    "Tanggal": pd.to_datetime(item["at"]),
                                }
                            )

                        df = pd.DataFrame(data_hasil)
                        st.session_state.df = df
                        st.session_state.info_app = info_app
                        st.success("✅ Analisis selesai! Klik tab **Dashboard** / **Data** di atas.")
                except Exception as e:
                    st.error(f"Terjadi Kesalahan: {e}")

# =========================================================
# TAB DASHBOARD
# =========================================================
with tab_dashboard:
    df = st.session_state.df
    info_app = st.session_state.info_app

    if df is None or info_app is None:
        st.info("Belum ada hasil. Silakan analisis dulu di tab **📌 Analisis**.")
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c_img, c_info, c_badge = st.columns([0.18, 0.62, 0.2], vertical_alignment="center")
        with c_img:
            st.image(info_app["icon"], width=72)
        with c_info:
            st.markdown(f"### {info_app['title']}")
            st.caption(f"{info_app['developer']}  •  ⭐ {info_app['score']}")
        with c_badge:
            st.markdown(
                f"""
                <div class="card card-tight">
                  <div class="muted" style="font-size:12px;">Total review dianalisis</div>
                  <div style="font-size:22px; font-weight:800;">{len(df)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")

        tot = len(df)
        pos = len(df[df["Sentimen"] == "Positif"])
        neg = len(df[df["Sentimen"] == "Negatif"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total", tot)
        m2.metric("Positif", pos, f"{(pos/tot*100):.1f}%")
        m3.metric("Negatif", neg, f"-{(neg/tot*100):.1f}%", delta_color="inverse")
        m4.metric("Skor Sentimen", round((pos - neg) / tot * 100, 1), "pos - neg (%)")

        st.write("")
        g1, g2 = st.columns(2)

        with g1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Perbandingan Sentimen")

            base = alt.Chart(df).encode(theta=alt.Theta("count()", stack=True))
            pie = base.mark_arc(innerRadius=65).encode(
                color=alt.Color("Sentimen", scale=alt.Scale(domain=["Positif", "Negatif"], range=[T["success"], T["danger"]])),
                tooltip=["Sentimen", "count()"],
            ).properties(height=320)
            st.altair_chart(pie, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with g2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Tren Harian")

            harian = df.groupby([pd.Grouper(key="Tanggal", freq="D"), "Sentimen"]).size().reset_index(name="Jumlah")
            line = alt.Chart(harian).mark_line(point=True).encode(
                x=alt.X("Tanggal:T", title="Tanggal"),
                y=alt.Y("Jumlah:Q", title="Jumlah Review"),
                color=alt.Color("Sentimen", scale=alt.Scale(domain=["Positif", "Negatif"], range=[T["success"], T["danger"]])),
                tooltip=["Tanggal:T", "Sentimen", "Jumlah:Q"],
            ).properties(height=320)
            st.altair_chart(line, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        w1, w2 = st.columns(2)
        with w1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 😊 WordCloud Positif")
            tp = " ".join(df[df["Sentimen"] == "Positif"]["Review Bersih"])
            if tp:
                st.pyplot(buat_wordcloud(tp))
            else:
                st.info("Tidak ada data positif.")
            st.markdown("</div>", unsafe_allow_html=True)

        with w2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 😡 WordCloud Negatif")
            tn = " ".join(df[df["Sentimen"] == "Negatif"]["Review Bersih"])
            if tn:
                st.pyplot(buat_wordcloud(tn))
            else:
                st.info("Tidak ada data negatif.")
            st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB DATA
# =========================================================
with tab_data:
    df = st.session_state.df
    info_app = st.session_state.info_app

    if df is None or info_app is None:
        st.info("Belum ada hasil. Silakan analisis dulu di tab **📌 Analisis**.")
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📝 Data Hasil Analisis")
        st.caption("Gunakan pencarian/scroll pada tabel. Kamu juga bisa download CSV.")
        st.dataframe(df, use_container_width=True, height=420)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            csv,
            "data_review_sentimen.csv",
            "text/csv",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
