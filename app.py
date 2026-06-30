"""
app.py — Streamlit App: Indonesia House Price Prediction
=======================================================
Run with: streamlit run app.py

Make sure you have run train_model.py first
so that the model/ folder contains pipeline.pkl, kota_list.pkl, and metadata.pkl.
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
    page_title="Indonesia House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── LOAD ARTIFACTS ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load model pipeline (cached to avoid reloading on each interaction)."""
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
def format_rupiah(value: float) -> str:
    if value >= 1e9:
        return f"Rp {value/1e9:.2f} Billion"
    elif value >= 1e6:
        return f"Rp {value/1e6:.1f} Million"
    else:
        return f"Rp {value:,.0f}"


def predict_price(kota, luas_tanah, luas_bangunan, kamar_tidur, kamar_mandi, garasi):
    """Generate prediction from user input."""
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
    """Provide an estimated range of ±15%."""
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
st.title("🏠 Indonesia House Price Prediction")
st.markdown("Estimate property prices based on housing features using a **Gradient Boosting Regressor** model.")

if pipeline is None:
    st.error(
        "⚠️ Model has not been trained yet. Please run: `python train_model.py` first.\n\n"
        "Ensure the dataset file exists at `data/house_prices.csv`"
    )
    st.stop()

# ─── SIDEBAR: METRIC SUMMARY ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Model Info")
    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | Training Data | {metadata['n_data']:,} houses |
    | R² Score | {metadata['metrics']['r2']:.3f} |
    | MAE | Rp {metadata['metrics']['mae']/1e6:.1f} M |
    | RMSE | Rp {metadata['metrics']['rmse']/1e6:.1f} M |
    """)

    st.markdown("---")
    st.markdown("### 📌 Dataset Price Range")
    st.markdown(f"""
    - **Min:** {format_rupiah(metadata['harga_min'])}
    - **Median:** {format_rupiah(metadata['harga_median'])}
    - **Max:** {format_rupiah(metadata['harga_max'])}
    """)

    st.markdown("---")
    st.caption("Developed by: **Kevinz Adhi Anggoro**")
    st.caption("Dataset: Kaggle — Harga Rumah Jabodetabek")
    st.caption("Model: Gradient Boosting | sklearn")

# ─── MAIN: INPUT FORM ─────────────────────────────────────────────────────────
st.markdown('<p class="section-header">📝 Enter Property Details</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    kota = st.selectbox(
        "📍 City / Region",
        options=kota_list,
        index=0,
        help="Select the city where the property is located"
    )
    luas_tanah = st.number_input(
        "📐 Land Area (sqm)",
        min_value=10, max_value=5000,
        value=120, step=10,
    )

with col2:
    luas_bangunan = st.number_input(
        "🏗️ Building Area (sqm)",
        min_value=10, max_value=3000,
        value=90, step=10,
    )
    kamar_tidur = st.slider(
        "🛏️ Number of Bedrooms",
        min_value=1, max_value=15, value=3
    )

with col3:
    kamar_mandi = st.slider(
        "🚿 Number of Bathrooms",
        min_value=1, max_value=10, value=2
    )
    garasi = st.slider(
        "🚗 Garage Capacity",
        min_value=0, max_value=8, value=1
    )

# Validation
if luas_bangunan > luas_tanah:
    st.warning("⚠️ Building area is larger than land area. Please ensure the data input is correct.")

st.markdown("---")

# ─── PREDICTION ─────────────────────────────────────────────────────────────────
col_btn, _ = st.columns([1, 3])
with col_btn:
    predict_btn = st.button("🔮 Predict Price", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner("Calculating prediction..."):
        harga_pred = predict_price(
            kota, luas_tanah, luas_bangunan,
            kamar_tidur, kamar_mandi, garasi
        )
        harga_min, harga_max = get_price_range(harga_pred)

    st.markdown("---")
    st.markdown('<p class="section-header">💰 Prediction Result</p>', unsafe_allow_html=True)

    res1, res2, res3 = st.columns(3)

    with res1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.85rem; color:#6c757d; margin-bottom:6px">Estimated Price</div>
            <div class="price-display">{format_rupiah(harga_pred)}</div>
            <div class="price-range">Range: {format_rupiah(harga_min)} – {format_rupiah(harga_max)}</div>
        </div>
        """, unsafe_allow_html=True)

    with res2:
        harga_per_m2 = harga_pred / luas_bangunan
        st.metric(
            label="Price per sqm of Building",
            value=format_rupiah(harga_per_m2),
        )

    with res3:
        harga_per_lt = harga_pred / luas_tanah
        st.metric(
            label="Price per sqm of Land",
            value=format_rupiah(harga_per_lt),
        )

    # ── Visualization: Gauge chart ───────────────────────────────────────────
    st.markdown("### 📈 Price Position within Dataset")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=harga_pred / 1e6,
        number={"suffix": " M", "valueformat": ".0f"},
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
                "ticksuffix": "M",
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
        title={"text": "Predicted Price (Million Rp)"},
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=60, b=20, l=20, r=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.caption(
        f"🔴 Red line = dataset median ({format_rupiah(metadata['harga_median'])}). "
        "Green = below median, Yellow = near median, Red = above median."
    )

    # ── Detail properti yang diinput ──────────────────────────────────────
    with st.expander("🔍 View Inputted Property Details"):
        detail = {
            "City": kota,
            "Land Area": f"{luas_tanah} sqm",
            "Building Area": f"{luas_bangunan} sqm",
            "Bedrooms": kamar_tidur,
            "Bathrooms": kamar_mandi,
            "Garages": garasi,
            "Building/Land Ratio": f"{luas_bangunan/luas_tanah:.2f}",
        }
        st.table(pd.DataFrame(detail.items(), columns=["Feature", "Value"]))

# ─── TAB: ANALISIS HARGA PER KOTA ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-header">🗺️ Exploration: Price Simulation per City</p>', unsafe_allow_html=True)
st.caption("Compare estimated prices across different cities with the exact same specifications.")

sim_col1, sim_col2, sim_col3 = st.columns(3)
with sim_col1:
    sim_lt = st.number_input("Land Area (sqm)", value=120, step=10, key="sim_lt")
with sim_col2:
    sim_lb = st.number_input("Building Area (sqm)", value=90, step=10, key="sim_lb")
with sim_col3:
    sim_kt = st.slider("Bedrooms", 1, 10, 3, key="sim_kt")

if st.button("📊 Compare All Cities", use_container_width=False):
    with st.spinner("Calculating for all cities..."):
        sim_results = []
        for k in kota_list:
            try:
                p = predict_price(k, sim_lt, sim_lb, sim_kt, 2, 1)
                sim_results.append({"City": k, "Estimated Price (Millions)": round(p / 1e6, 1)})
            except Exception:
                pass

        df_sim = pd.DataFrame(sim_results).sort_values("Estimated Price (Millions)", ascending=False)

    fig_bar = px.bar(
        df_sim.head(20),
        x="Estimated Price (Millions)",
        y="City",
        orientation="h",
        color="Estimated Price (Millions)",
        color_continuous_scale="Blues",
        title=f"Top 20 Cities · Land={sim_lt}sqm · Building={sim_lb}sqm · {sim_kt} BR",
        labels={"Estimated Price (Millions)": "Estimated Price (Million Rp)"},
    )
    fig_bar.update_layout(
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        height=500,
        margin=dict(l=10, r=10, t=50, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.dataframe(
        df_sim.style.background_gradient(subset=["Estimated Price (Millions)"], cmap="Blues"),
        use_container_width=True,
        hide_index=True,
    )