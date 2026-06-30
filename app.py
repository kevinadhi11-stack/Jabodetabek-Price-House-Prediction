"""
app.py — Streamlit App: Prediksi Harga Rumah Indonesia
=======================================================
Jalankan dengan: streamlit run app.py

Pastikan sudah menjalankan train_model.py terlebih dahulu
sehingga folder model/ berisi pipeline.pkl, kota_list.pkl, metadata.pkl.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Harga Rumah Indonesia",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── LOAD ARTIFACTS ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load model pipeline (di-cache agar tidak reload tiap interaksi)."""
    model_dir = Path("model")
    if not model_dir.exists():
        return None, None, None

    with open(model_dir / "pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    with open(model_dir / "kota_list.pkl", "rb") as f:
        kota_list = pickle.load(f)
    with open(model_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    return pipeline, kota_list, metadata


pipeline, kota_list, metadata = load_model()

# ─── HELPER ───────────────────────────────────────────────────────────────────
def format_rupiah(nilai: float) -> str:
    if nilai >= 1e9:
        return f"Rp {nilai/1e9:.2f} Miliar"
    elif nilai >= 1e6:
        return f"Rp {nilai/1e6:.1f} Juta"
    else:
        return f"Rp {nilai:,.0f}"


def predict_price(kota, luas_tanah, luas_bangunan, kamar_tidur, kamar_mandi, garasi):
    """Buat prediksi dari input user."""
    rasio_bangunan = luas_bangunan / (luas_tanah + 1)
    total_kamar    = kamar_tidur + kamar_mandi

    input_df = pd.DataFrame([{
        "luas_tanah":    luas_tanah,
        "luas_bangunan": luas_bangunan,
        "kamar_tidur":   kamar_tidur,
        "kamar_mandi":   kamar_mandi,
        "garasi":        garasi,
        "rasio_bangunan":rasio_bangunan,
        "total_kamar":   total_kamar,
        "kota":          kota,
    }])

    log_pred = pipeline.predict(input_df)[0]
    return np.expm1(log_pred)


def get_price_range(harga_pred: float):
    """Beri rentang estimasi ±15%."""
    return harga_pred * 0.85, harga_pred * 1.15


# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        border: 1px solid #e9ecef;
    }
    .price-display {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a73e8;
        line-height: 1.2;
    }
    .price-range {
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: 4px;
    }
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #343a40;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #1a73e8;
    }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.title("🏠 Prediksi Harga Rumah Indonesia")
st.markdown("Estimasi harga properti berdasarkan fitur rumah menggunakan **Gradient Boosting Regressor**.")

if pipeline is None:
    st.error(
        "⚠️ Model belum dilatih. Jalankan dulu: `python train_model.py`\n\n"
        "Pastikan file dataset ada di `data/harga_rumah.csv`"
    )
    st.stop()

# ─── SIDEBAR: METRIC SUMMARY ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Info Model")
    st.markdown(f"""
    | Metrik | Nilai |
    |--------|-------|
    | Data training | {metadata['n_data']:,} rumah |
    | R² Score | {metadata['metrics']['r2']:.3f} |
    | MAE | Rp {metadata['metrics']['mae']/1e6:.1f} juta |
    | RMSE | Rp {metadata['metrics']['rmse']/1e6:.1f} juta |
    """)

    st.markdown("---")
    st.markdown("### 📌 Rentang Harga Dataset")
    st.markdown(f"""
    - **Min:** {format_rupiah(metadata['harga_min'])}
    - **Median:** {format_rupiah(metadata['harga_median'])}
    - **Maks:** {format_rupiah(metadata['harga_max'])}
    """)

    st.markdown("---")
    st.caption("Dibuat oleh: **[Kevinz Adhi]**")
    st.caption("Dataset: Kaggle — Harga Rumah Jabodetabek")
    st.caption("Model: Gradient Boosting | sklearn")

# ─── MAIN: INPUT FORM ─────────────────────────────────────────────────────────
st.markdown('<p class="section-header">📝 Masukkan Detail Properti</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    kota = st.selectbox(
        "📍 Kota / Wilayah",
        options=kota_list,
        index=0,
        help="Pilih kota lokasi properti"
    )
    luas_tanah = st.number_input(
        "📐 Luas Tanah (m²)",
        min_value=10, max_value=5000,
        value=120, step=10,
    )

with col2:
    luas_bangunan = st.number_input(
        "🏗️ Luas Bangunan (m²)",
        min_value=10, max_value=3000,
        value=90, step=10,
    )
    kamar_tidur = st.slider(
        "🛏️ Jumlah Kamar Tidur",
        min_value=1, max_value=15, value=3
    )

with col3:
    kamar_mandi = st.slider(
        "🚿 Jumlah Kamar Mandi",
        min_value=1, max_value=10, value=2
    )
    garasi = st.slider(
        "🚗 Kapasitas Garasi",
        min_value=0, max_value=8, value=1
    )

# Validasi
if luas_bangunan > luas_tanah:
    st.warning("⚠️ Luas bangunan lebih besar dari luas tanah. Pastikan data sudah benar.")

st.markdown("---")

# ─── PREDIKSI ─────────────────────────────────────────────────────────────────
col_btn, _ = st.columns([1, 3])
with col_btn:
    predict_btn = st.button("🔮 Prediksi Harga", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner("Menghitung prediksi..."):
        harga_pred = predict_price(
            kota, luas_tanah, luas_bangunan,
            kamar_tidur, kamar_mandi, garasi
        )
        harga_min, harga_max = get_price_range(harga_pred)

    st.markdown("---")
    st.markdown('<p class="section-header">💰 Hasil Prediksi</p>', unsafe_allow_html=True)

    res1, res2, res3 = st.columns(3)

    with res1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.85rem; color:#6c757d; margin-bottom:6px">Estimasi Harga</div>
            <div class="price-display">{format_rupiah(harga_pred)}</div>
            <div class="price-range">Rentang: {format_rupiah(harga_min)} – {format_rupiah(harga_max)}</div>
        </div>
        """, unsafe_allow_html=True)

    with res2:
        harga_per_m2 = harga_pred / luas_bangunan
        st.metric(
            label="Harga per m² Bangunan",
            value=format_rupiah(harga_per_m2),
        )

    with res3:
        harga_per_lt = harga_pred / luas_tanah
        st.metric(
            label="Harga per m² Tanah",
            value=format_rupiah(harga_per_lt),
        )

    # ── Visualisasi: Gauge chart ───────────────────────────────────────────
    st.markdown("### 📈 Posisi Harga di Dataset")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=harga_pred / 1e6,
        number={"suffix": " Juta", "valueformat": ".0f"},
        delta={
            "reference": metadata["harga_median"] / 1e6,
            "relative": True,
            "valueformat": ".1%",
            "suffix": " vs median",
        },
        gauge={
            "axis": {
                "range": [
                    metadata["harga_min"] / 1e6,
                    min(metadata["harga_max"] / 1e6, harga_pred * 3 / 1e6)
                ],
                "tickformat": ".0f",
                "ticksuffix": "Jt",
            },
            "bar": {"color": "#1a73e8"},
            "steps": [
                {"range": [metadata["harga_min"] / 1e6, metadata["harga_median"] * 0.7 / 1e6], "color": "#d4edda"},
                {"range": [metadata["harga_median"] * 0.7 / 1e6, metadata["harga_median"] * 1.3 / 1e6], "color": "#fff3cd"},
                {"range": [metadata["harga_median"] * 1.3 / 1e6, min(metadata["harga_max"] / 1e6, harga_pred * 3 / 1e6)], "color": "#f8d7da"},
            ],
            "threshold": {
                "line": {"color": "#dc3545", "width": 3},
                "thickness": 0.75,
                "value": metadata["harga_median"] / 1e6,
            },
        },
        title={"text": "Harga Prediksi (Rupiah)"},
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=60, b=20, l=20, r=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.caption(
        f"🔴 Garis merah = median dataset ({format_rupiah(metadata['harga_median'])}). "
        "Hijau = di bawah median, Kuning = sekitar median, Merah = di atas median."
    )

    # ── Detail properti yang diinput ──────────────────────────────────────
    with st.expander("🔍 Lihat detail properti yang diinput"):
        detail = {
            "Kota": kota,
            "Luas Tanah": f"{luas_tanah} m²",
            "Luas Bangunan": f"{luas_bangunan} m²",
            "Kamar Tidur": kamar_tidur,
            "Kamar Mandi": kamar_mandi,
            "Garasi": garasi,
            "Rasio Bangunan/Tanah": f"{luas_bangunan/luas_tanah:.2f}",
        }
        st.table(pd.DataFrame(detail.items(), columns=["Fitur", "Nilai"]))

# ─── TAB: ANALISIS HARGA PER KOTA ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-header">🗺️ Eksplorasi: Simulasi Harga per Kota</p>', unsafe_allow_html=True)
st.caption("Bandingkan estimasi harga untuk berbagai kota dengan spesifikasi yang sama.")

sim_col1, sim_col2, sim_col3 = st.columns(3)
with sim_col1:
    sim_lt = st.number_input("Luas Tanah (m²)", value=120, step=10, key="sim_lt")
with sim_col2:
    sim_lb = st.number_input("Luas Bangunan (m²)", value=90, step=10, key="sim_lb")
with sim_col3:
    sim_kt = st.slider("Kamar Tidur", 1, 10, 3, key="sim_kt")

if st.button("📊 Bandingkan Semua Kota", use_container_width=False):
    with st.spinner("Menghitung untuk semua kota..."):
        sim_results = []
        for k in kota_list:
            try:
                p = predict_price(k, sim_lt, sim_lb, sim_kt, 2, 1)
                sim_results.append({"Kota": k, "Estimasi Harga (Juta)": round(p / 1e6, 1)})
            except Exception:
                pass

        df_sim = pd.DataFrame(sim_results).sort_values("Estimasi Harga (Juta)", ascending=False)

    fig_bar = px.bar(
        df_sim.head(20),
        x="Estimasi Harga (Juta)",
        y="Kota",
        orientation="h",
        color="Estimasi Harga (Juta)",
        color_continuous_scale="Blues",
        title=f"Top 20 Kota · LT={sim_lt}m² · LB={sim_lb}m² · {sim_kt} KT",
        labels={"Estimasi Harga (Juta)": "Harga Estimasi (Juta Rp)"},
    )
    fig_bar.update_layout(
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        height=500,
        margin=dict(l=10, r=10, t=50, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.dataframe(
        df_sim.style.background_gradient(subset=["Estimasi Harga (Juta)"], cmap="Blues"),
        use_container_width=True,
        hide_index=True,
    )
