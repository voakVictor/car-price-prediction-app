import streamlit as st
import numpy as np
import pickle
import os

# --------------------------------------------------------
#  PAGE CONFIG
# --------------------------------------------------------
st.set_page_config(
    page_title="Car Price Prediction System",
    page_icon="🚗",
    layout="centered"
)

# --------------------------------------------------------
#  CUSTOM BACKGROUND IMAGE
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
#  CUSTOM UI STYLES
# --------------------------------------------------------
st.markdown("""
    <style>

        .title {
            font-size: 42px;
            font-weight: 900;
            color: #ffffff;
            text-shadow: 2px 2px 4px #000000;
            text-align: center;
            margin-bottom: 10px;
        }

        .subtitle {
            font-size: 18px;
            color: #f5f5f5;
            text-align: center;
            margin-bottom: 30px;
        }

        .card {
            background: rgba(255, 255, 255, 0.15);
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
            backdrop-filter: blur(10px);
        }

        .prediction-box {
            background-color: #0d6efd;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            color: white;
            margin-top: 20px;
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
    st.error("❌ Model file not found. Make sure 'car_price_model.pkl' is uploaded.")
    st.stop()

model = pickle.load(open(MODEL_PATH, "rb"))

scaler = None
if os.path.exists(SCALER_PATH):
    scaler = pickle.load(open(SCALER_PATH, "rb"))
else:
    st.warning("⚠️ Scaler file not found — predictions may be less accurate.")

# --------------------------------------------------------
#  INPUT FORM (Inside Card)
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

# Convert categorical values
leather_interior = 1 if leather_interior == "Yes" else 0

input_data = np.array([[prod_year, leather_interior, engine_volume, mileage, cylinders]])

# Apply scaling
if scaler is not None:
    try:
        input_data = scaler.transform(input_data)
    except Exception as e:
        st.error(f"Scaling error: {e}")
        st.stop()

# --------------------------------------------------------
#  PREDICTION BUTTON
# --------------------------------------------------------
if st.button("Predict Price ☑️"):
    try:
        prediction = model.predict(input_data)
        predicted_price = float(prediction[0])

        st.markdown(
            f'<div class="prediction-box">Estimated Price:<br> GHS {predicted_price:,.2f}</div>',
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --------------------------------------------------------
#  FOOTER
# --------------------------------------------------------
st.markdown("""
    <hr>
    <div style='text-align:center; color:white; font-size:16px;'>
        Built by <strong>Victor Kwabena Opare-Addo</strong> • Powered by Streamlit 🚀
    </div>
""", unsafe_allow_html=True)
