import streamlit as st
import numpy as np
import pickle
import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
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
# BACKGROUND IMAGE (DARK, NON-WHITE, DIRECT LINK)
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
# UI STYLES
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
    text-shadow: 2px 2px 6px black;
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

**Features used**
- Production Year
- Engine Volume
- Mileage
- Leather Interior
- Cylinders

**Model**
- Linear Regression
- Scaled inputs

**Developer**
Victor Kwabena Opare‑Addo

**Version**
v1.1 (PDF Report Added)
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
    st.error("❌ Model file not found.")
    st.stop()

model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb")) if os.path.exists(SCALER_PATH) else None

# --------------------------------------------------------
# TOP PREDICTION BOX
# --------------------------------------------------------
prediction_box = st.markdown(
    '<div class="top-card">Enter car details below</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------------
# INPUT FORM
# --------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="small")

with col1:
    prod_year = st.number_input("Production Year", 1900, 2026, 2015)
    engine_volume = st.number_input("Engine Volume (Liters)", 0.5, 10.0, 2.0, step=0.1)

with col2:
    leather = st.selectbox("Leather Interior", ["No", "Yes"])
    mileage = st.number_input("Mileage (KM)", 0, 1_000_000, 50000)

cylinders = st.number_input("Number of Cylinders", 1, 16, 4)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------
# PROCESS INPUT
# --------------------------------------------------------
leather_val = 1 if leather == "Yes" else 0
input_data = np.array([[prod_year, leather_val, engine_volume, mileage, cylinders]])

if scaler is not None:
    input_data = scaler.transform(input_data)

# --------------------------------------------------------
# PDF GENERATOR FUNCTION (OPTION A)
# --------------------------------------------------------
def generate_pdf(data, price):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Car Price Prediction Report")

    c.setFont("Helvetica", 12)
    y = height - 100

    for label, value in data.items():
        c.drawString(50, y, f"{label}: {value}")
        y -= 25

    y -= 20
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, f"Estimated Price: GHS {price:,.2f}")

    y -= 40
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, y - 15, "Generated by Car Price Prediction System")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# --------------------------------------------------------
# PREDICT BUTTON
# --------------------------------------------------------
if st.button("Predict Price ☑️"):
    price = float(model.predict(input_data)[0])

    prediction_box.markdown(
        f'<div class="top-card">Estimated Price: GHS {price:,.2f}</div>',
        unsafe_allow_html=True
    )

    report_data = {
        "Production Year": prod_year,
        "Engine Volume (L)": engine_volume,
        "Mileage (KM)": mileage,
        "Leather Interior": leather,
        "Cylinders": cylinders
    }

    pdf_file = generate_pdf(report_data, price)

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
<div style='text-align:center;color:white;text-shadow:2px 2px 6px black;'>
Built by <strong>Victor Kwabena Opare‑Addo</strong> • Powered by Streamlit 🚀
</div>
""", unsafe_allow_html=True)
