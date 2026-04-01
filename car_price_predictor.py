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
# BACKGROUND + UI STYLING (FULLY RESPONSIVE)
# --------------------------------------------------------
def apply_styles():
    st.markdown(
        """
        <style>

        /* Global Page Padding Fix */
        .stApp {
            background-image: url("https://images.pexels.com/photos/919073/pexels-photo-919073.jpeg");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }

        .block-container {
            padding-top: 2rem !important;
            max-width: 760px !important;
        }

        /* Title */
        .title-text {
            font-size: 36px;
            font-weight: 900;
            color: #ffffff;
            text-align: center;
            margin-bottom: 4px;
            text-shadow: 3px 3px 10px rgba(0,0,0,1);
        }

        /* Subtitle */
        .subtitle-text {
            font-size: 17px;
            font-weight: 500;
            color: #f0f0f0;
            text-align: center;
            margin-bottom: 1.5rem;
            text-shadow: 2px 2px 6px rgba(0,0,0,1);
        }

        /* Prediction box */
        .top-card {
            background: rgba(13,110,253,0.92);
            padding: 15px;
            border-radius: 12px;
            font-size: 20px;
            color: white;
            font-weight: 700;
            text-align: center;
            box-shadow: 0px 6px 25px rgba(0,0,0,0.9);
            margin-bottom: 1.2rem;
        }

        /* Input card */
        .input-card {
            background: rgba(0,0,0,0.60);
            padding: 20px;
            border-radius: 14px;
            box-shadow: 0px 6px 25px rgba(0,0,0,0.9);
            margin-bottom: 1.5rem;
        }

        /* Field labels */
        label {
            color: white !important;
            font-size: 16px !important;
            font-weight: 700 !important;
            text-shadow: 2px 2px 8px black !important;
        }

        /* Input fields */
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] {
            background-color: rgba(255,255,255,0.22) !important;
            color: white !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255,255,255,0.35) !important;
        }

        /* Mobile responsiveness */
        @media (max-width: 600px) {
            .title-text {
                font-size: 28px !important;
            }
            .subtitle-text {
                font-size: 14px !important;
            }
            .top-card {
                font-size: 18px !important;
                padding: 12px !important;
            }
        }

        </style>
        """,
        unsafe_allow_html=True
    )

apply_styles()

# --------------------------------------------------------
# SIDEBAR INFO
# --------------------------------------------------------
st.sidebar.title("ℹ️ App Information")
st.sidebar.markdown("""
### 📘 Purpose
Academic demonstration of a car price prediction system.

### 🔍 Features Used
- Production Year  
- Leather Interior  
- Engine Volume  
- Mileage  
- Cylinders  

### 🧠 Model
Trained Linear Regression model.

### 👨‍💻 Developer
**Victor Kwabena Opare‑Addo**

### 🎓 Version
Academic Edition 1.0
""")

# --------------------------------------------------------
# PAGE HEADER
# --------------------------------------------------------
st.markdown('<div class="title-text">🚗 Car Price Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">A clean and responsive interface for academic learning</div>', unsafe_allow_html=True)

# --------------------------------------------------------
# LOAD MODEL & SCALER
# --------------------------------------------------------
MODEL_PATH = "car_price_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file 'car_price_model.pkl' is missing.")
    st.stop()

model = pickle.load(open(MODEL_PATH, "rb"))

scaler = pickle.load(open(SCALER_PATH, "rb")) if os.path.exists(SCALER_PATH) else None

# --------------------------------------------------------
# TOP RESULT PLACEHOLDER
# --------------------------------------------------------
prediction_box = st.markdown(
    '<div class="top-card">Enter details below to generate prediction</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------------
# INPUT FORM (RESPONSIVE)
# --------------------------------------------------------
st.markdown('<div class="input-card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Production Year", min_value=1900, max_value=2026, value=2015)
    engine = st.number_input("Engine Volume (Liters)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)

with col2:
    leather = st.selectbox("Leather Interior", ["No", "Yes"])
    mileage = st.number_input("Mileage (KM)", min_value=0, max_value=1_000_000, value=50000)

cyl = st.number_input("Number of Cylinders", min_value=1, max_value=16, value=4)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------
# PROCESS INPUT
# --------------------------------------------------------
leather_val = 1 if leather == "Yes" else 0

input_data = np.array([[year, leather_val, engine, mileage, cyl]])

if scaler:
    input_data = scaler.transform(input_data)

# --------------------------------------------------------
# PREDICT BUTTON
# --------------------------------------------------------
if st.button("Predict Price"):
    pred = float(model.predict(input_data)[0])

    prediction_box.markdown(
        f'<div class="top-card">Estimated Price: GHS {pred:,.2f}</div>',
        unsafe_allow_html=True
    )

# --------------------------------------------------------
# FOOTER
# --------------------------------------------------------
st.markdown("""
<hr style="border:1px solid white;">
<div style='text-align:center;color:white;text-shadow:2px 2px 6px black;'>
Made for Academic Learning • Streamlit Project • By <b>Victor Opare‑Addo</b>
</div>
""", unsafe_allow_html=True)
