import streamlit as st
import numpy as np
import pickle
import os

# --------------------------------------------------------
#  PAGE CONFIGURATION
# --------------------------------------------------------
st.set_page_config(
    page_title="Car Price Prediction System",
    page_icon="🚗",
    layout="centered"
)

# --------------------------------------------------------
#  BACKGROUND IMAGE
# --------------------------------------------------------
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1503376780353-7e6692767b70");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()

# --------------------------------------------------------
#  CUSTOM CSS FOR VISIBILITY & READABILITY
# --------------------------------------------------------
st.markdown("""
    <style>

        /* ---- Title ---- */
        .title {
            font-size: 42px;
            font-weight: 900;
            color: #ffffff;
            text-shadow: 3px 3px 8px rgba(0,0,0,0.9);
            text-align: center;
            margin-bottom: 5px;
        }

        /* ---- Subtitle ---- */
        .subtitle {
            font-size: 18px;
            color: #e6e6e6;
            text-align: center;
            margin-bottom: 30px;
            text-shadow: 2px 2px 6px rgba(0,0,0,0.9);
        }

        /* ---- Glass Card for Form ---- */
        .card {
            background: rgba(255, 255, 255, 0.35); /* increased opacity */
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0px 6px 28px rgba(0,0,0,0.7);
            backdrop-filter: blur(14px);
        }

        /* ---- Input Labels ---- */
        label, .stNumberInput label, .stSelectbox label {
            color: #ffffff !important;
            font-weight: 700 !important;
            text-shadow: 2px 2px 5px rgba(0,0,0,1);
            font-size: 16px !important;
        }

        /* ---- Input Fields ---- */
        .stNumberInput input {
            background-color: rgba(0, 0, 0, 0.65) !important;
            color: #ffffff !important;
            border-radius: 8px !important;
            padding: 8px !important;
        }

        /* ---- Selectbox ---- */
        .stSelectbox div[data-baseweb="select"] {
            background-color: rgba(0, 0, 0, 0.65) !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 6px !important;
        }

        /* ---- Dropdown Options ---- */
        .stSelectbox div[role="listbox"] {
            background-color: rgba(30, 30, 30, 0.95) !important;
            color: white !important;
        }

        /* ---- Prediction Box ---- */
        .prediction-box {
            background-color: #0069ff;
            padding: 22px;
            border-radius: 12px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: white;
            margin-top: 25px;
            box-shadow: 0px 6px 25px rgba(0,0,0,0.8);
        }

    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
#  HEADERS
# --------------------------------------------------------
st.markdown('<div class="title">🚗 Car Price Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter your car details below to get an instant valuation</div>', unsafe_allow_html=True)

# --------------------------------------------------------
#  LOAD MODEL & SCALER
# --------------------------------------------------------
MODEL_PATH = "car_price_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Ensure 'car_price_model.pkl' is in the folder.")
    st.stop()

model = pickle.load(open(MODEL_PATH, "rb"))

scaler = None
if os.path.exists(SCALER_PATH):
    scaler = pickle.load(open(SCALER_PATH, "rb"))
else:
    st.warning("⚠️ Scaler not found. Predictions may be less accurate.")

# --------------------------------------------------------
#  INPUT FORM (INSIDE GLASS CARD)
# --------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    prod_year = st.number_input("Production Year", min_value=1900, max_value=2026, value=2015)
    engine_volume = st.number_input("Engine Volume (Liters)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)

with col2:
    leather_interior = st.selectbox("Leather Interior", ["No", "Yes"])
    mileage = st.number_input("Mileage (KM)", min_value=0, max_value=1000000, value=50000)

cylinders = st.number_input("Number of Cylinders", min_value=1, max_value=16, value=4)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------
#  DATA PROCESSING
# --------------------------------------------------------
leather_interior = 1 if leather_interior == "Yes" else 0

input_data = np.array([[prod_year, leather_interior, engine_volume, mileage, cylinders]])

if scaler is not None:
    try:
        input_data = scaler.transform(input_data)
    except Exception as e:
        st.error(f"Error during scaling: {e}")
        st.stop()

# --------------------------------------------------------
#  PREDICTION BUTTON
# --------------------------------------------------------
if st.button("Predict Price ☑️"):
    try:
        prediction = model.predict(input_data)
        predicted_price = float(prediction[0])

        st.markdown(
            f'<div class="prediction-box">Estimated Price:<br>GHS {predicted_price:,.2f}</div>',
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --------------------------------------------------------
#  FOOTER
# --------------------------------------------------------
st.markdown("""
    <hr>
    <div style='text-align:center; color:white; font-size:16px; text-shadow: 2px 2px 5px black;'>
        Built by <strong>Victor Kwabena Opare‑Addo</strong> • Powered by Streamlit 🚀
    </div>
""", unsafe_allow_html=True)
