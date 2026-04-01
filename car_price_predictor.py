import streamlit as st
import numpy as np
import pickle
import os

# --------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------
st.set_page_config(
    page_title="Car Price Prediction System",
    page_icon="🚗",
    layout="centered"
)

# --------------------------------------------------------
# BACKGROUND IMAGE (DIRECT .JPG LINK)
# --------------------------------------------------------
def set_bg():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.pexels.com/photos/919073/pexels-photo-919073.jpeg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

# --------------------------------------------------------
# CSS FOR VISIBILITY & UI
# --------------------------------------------------------
st.markdown("""
    <style>

        .title {
            font-size: 40px;
            font-weight: 900;
            color: white;
            text-shadow: 3px 3px 10px black;
            text-align: center;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 18px;
            color: #f0f0f0;
            text-shadow: 2px 2px 8px black;
            text-align: center;
            margin-bottom: 25px;
        }

        .card {
            background: rgba(0, 0, 0, 0.55);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 6px 30px rgba(0,0,0,0.9);
            backdrop-filter: blur(10px);
        }

        .top-card {
            background: rgba(13, 110, 253, 0.9);
            padding: 20px;
            border-radius: 15px;
            color: white;
            font-size: 22px;
            font-weight: bold;
            text-align: center;
            box-shadow: 0 6px 30px rgba(0,0,0,0.8);
            margin-bottom: 20px;
        }

        label {
            color: white !important;
            font-weight: 700 !important;
            text-shadow: 2px 2px 6px black;
        }

        .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
            background-color: rgba(255,255,255,0.2) !important;
            color: white !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255,255,255,0.5);
        }

    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# HEADERS
# --------------------------------------------------------
st.markdown('<div class="title">🚗 Car Price Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter your car details below to get an instant valuation</div>', unsafe_allow_html=True)

# --------------------------------------------------------
# LOAD MODEL & SCALER
# --------------------------------------------------------
MODEL_PATH = "car_price_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not found. Upload 'car_price_model.pkl'.")
    st.stop()

model = pickle.load(open(MODEL_PATH, "rb"))

scaler = None
if os.path.exists(SCALER_PATH):
    scaler = pickle.load(open(SCALER_PATH, "rb"))
else:
    st.warning("⚠️ Scaler not found — predictions may be less accurate.")

# --------------------------------------------------------
# TOP PREDICTION BOX (INITIAL EMPTY)
# --------------------------------------------------------
prediction_placeholder = st.markdown(
    '<div class="top-card">Enter details & click Predict</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------------
# INPUT FORM INSIDE A CARD
# --------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    prod_year = st.number_input("Production Year", 1900, 2026, 2015)
    engine_volume = st.number_input("Engine Volume (Liters)", 0.5, 10.0, 2.0, step=0.1)

with col2:
    leather_interior = st.selectbox("Leather Interior", ["No", "Yes"])
    mileage = st.number_input("Mileage (KM)", 0, 1000000, 50000)

cylinders = st.number_input("Number of Cylinders", 1, 16, 4)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------
# PROCESS INPUT
# --------------------------------------------------------
leather_interior = 1 if leather_interior == "Yes" else 0

input_data = np.array([[prod_year, leather_interior, engine_volume, mileage, cylinders]])

if scaler is not None:
    try:
        input_data = scaler.transform(input_data)
    except Exception as e:
        st.error(f"Scaler error: {e}")
        st.stop()

# --------------------------------------------------------
# PREDICT BUTTON
# --------------------------------------------------------
if st.button("Predict Price ☑️"):
    try:
        prediction = model.predict(input_data)
        price = float(prediction[0])

        # Update top card instead of showing below button
        prediction_placeholder.markdown(
            f'<div class="top-card">Estimated Price: GHS {price:,.2f}</div>',
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --------------------------------------------------------
# FOOTER
# --------------------------------------------------------
st.markdown("""
    <hr>
    <div style='text-align:center;color:white;text-shadow:2px 2px 6px black;'>
        Built by <b>Victor Kwabena Opare‑Addo</b> • Powered by Streamlit 🚀
    </div>
""", unsafe_allow_html=True)
