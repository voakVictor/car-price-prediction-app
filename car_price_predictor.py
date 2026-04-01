import streamlit as st
import numpy as np
import pickle
import os
from datetime import datetime
from fpdf import FPDF
import io

# --------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------
st.set_page_config(
    page_title="Car Price Prediction System",
    page_icon="🚗",
    layout="centered"
)

# --------------------------------------------------------
# BACKGROUND IMAGE
# --------------------------------------------------------
def set_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.pexels.com/photos/919073/pexels-photo-919073.jpeg");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            max-width: 760px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

# --------------------------------------------------------
# CLEAN UI CSS
# --------------------------------------------------------
st.markdown("""
<style>
.title {
    font-size: 34px;
    font-weight: 900;
    color: white;
    text-align: center;
    text-shadow: 3px 3px 10px black;
}
.subtitle {
    font-size: 16px;
    color: #efefef;
    text-align: center;
    margin-bottom: 1rem;
    text-shadow: 2px 2px 6px black;
}
.top-card {
    background: rgba(13,110,253,0.92);
    padding: 14px;
    border-radius: 12px;
    color: white;
    font-size: 20px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 1rem;
}
.card {
    background: rgba(0,0,0,0.55);
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 6px 25px rgba(0,0,0,0.9);
}
label {
    color: white !important;
    font-weight: 600 !important;
}
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background-color: rgba(255,255,255,0.18) !important;
    color: white !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255,255,255,0.4);
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# SIDEBAR INFO
# --------------------------------------------------------
st.sidebar.title("ℹ️ App Info")
st.sidebar.markdown("""
### 🚗 Car Price Predictor
Predicts car prices using a trained **machine learning model**.

### Features Used
- Production Year  
- Engine Volume  
- Mileage  
- Leather Interior  
- Cylinders  

### Developer
**Victor Kwabena Opare‑Addo**

### Version  
v1.2 — PDF Generator (fpdf2)
""")

# --------------------------------------------------------
# HEADER
# --------------------------------------------------------
st.markdown('<div class="title">🚗 Car Price Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Instant vehicle price estimation</div>', unsafe_allow_html=True)

# --------------------------------------------------------
# LOAD MODEL & SCALER
# --------------------------------------------------------
MODEL_PATH = "car_price_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file missing.")
    st.stop()

model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb")) if os.path.exists(SCALER_PATH) else None

# --------------------------------------------------------
# TOP PREDICTION BOX
# --------------------------------------------------------
prediction_box = st.markdown(
    '<div class="top-card">Enter details below</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------------
# INPUT CARD
# --------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="small")

with col1:
    prod_year = st.number_input("Production Year", 1900, 2026, 2015)
    engine_volume = st.number_input("Engine Volume (L)", 0.5, 10.0, 2.0, step=0.1)

with col2:
    leather = st.selectbox("Leather Interior", ["No", "Yes"])
    mileage = st.number_input("Mileage (KM)", 0, 1_000_000, 50000)

cylinders = st.number_input("Cylinders", 1, 16, 4)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------
# PROCESS INPUT
# --------------------------------------------------------
leather_val = 1 if leather == "Yes" else 0
data = np.array([[prod_year, leather_val, engine_volume, mileage, cylinders]])

if scaler is not None:
    data = scaler.transform(data)

# --------------------------------------------------------
# PDF GENERATOR (fpdf2 – STREAMLIT CLOUD SAFE)
# --------------------------------------------------------
def generate_pdf(info, price):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=14)

    pdf.cell(0, 10, "Car Price Prediction Report", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Helvetica", size=12)

    for label, value in info.items():
        pdf.cell(0, 10, f"{label}: {value}", ln=True)

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, f"Estimated Price: GHS {price:,.2f}", ln=True)

    pdf.ln(10)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# --------------------------------------------------------
# PREDICT BUTTON
# --------------------------------------------------------
if st.button("Predict Price ☑️"):
    price = float(model.predict(data)[0])

    prediction_box.markdown(
        f'<div class="top-card">Estimated Price: GHS {price:,.2f}</div>',
        unsafe_allow_html=True
    )

    info = {
        "Production Year": prod_year,
        "Engine Volume": f"{engine_volume} L",
        "Mileage": f"{mileage:,} KM",
        "Leather Interior": leather,
        "Cylinders": cylinders
    }

    pdf_file = generate_pdf(info, price)

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_file,
        file_name="car_price_report.pdf",
        mime="application/pdf"
    )

# --------------------------------------------------------
# FOOTER
# --------------------------------------------------------
st.markdown("""
<hr>
<div style='text-align:center;color:white;'>
Built by <strong>Victor Kwabena Opare‑Addo</strong> • Powered by Streamlit 🚀
</div>
""", unsafe_allow_html=True)
