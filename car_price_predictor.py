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
# BACKGROUND + GLOBAL STYLES
# --------------------------------------------------------
def apply_styles():
    st.markdown("""
        <style>

        /* --- Background Image --- */
        .stApp {
            background-image: url("https://images.pexels.com/photos/919073/pexels-photo-919073.jpeg");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }

        /* --- Main container spacing --- */
        .block-container {
            padding-top: 2rem !important;
            max-width: 760px !important;
        }

        /* --- Page Title --- */
        .title {
            font-size: 36px;
            font-weight: 900;
            color: white;
            text-align: center;
            margin-bottom: 4px;
            text-shadow: 3px 3px 10px rgba(0,0,0,1);
        }

        /* --- Subtitle --- */
        .subtitle {
            font-size: 16px;
            color: #ececec;
            text-align: center;
            margin-bottom: 1.4rem;
            text-shadow: 2px 2px 6px rgba(0,0,0,1);
        }

        /* --- Prediction Card --- */
        .top-card {
            background: rgba(13,110,253,0.92);
            padding: 15px;
            border-radius: 12px;
            color: white;
            text-align: center;
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 1.4rem;
            box-shadow: 0px 6px 25px rgba(0,0,0,0.85);
        }

        /* --- Input Form Card --- */
        .card {
            background: rgba(0,0,0,0.55);
            padding: 20px;
            border-radius: 14px;
            box-shadow: 0px 6px 25px rgba(0,0,0,0.9);
            backdrop-filter: blur(8px);
            margin-bottom: 1.6rem;
        }

        /* --- Input Labels --- */
        label {
            font-weight: 700 !important;
            color: white !important;
            font-size: 16px !important;
            text-shadow: 2px 2px 8px black !important;
        }

        /* --- Input Fields --- */
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] {
            background-color: rgba(255,255,255,0.18) !important;
            color: white !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255,255,255,0.35) !important;
        }

        /* --- Dropdown Menu --- */
        .stSelectbox div[role="listbox"] {
            background-color: rgba(15,15,15,0.95) !important;
            color: white !important;
        }

        /* --- Mobile Adjustments --- */
        @media(max-width: 600px) {
            .title { font-size: 28px !important; }
            .subtitle { font-size: 14px !important; }
            .top-card { font-size: 18px !important; }
        }

        </style>
    """, unsafe_allow_html=True)

apply_styles()

# --------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------
st.sidebar.title("ℹ️ App Info")
st.sidebar.markdown("""
### 🚗 Car Price Prediction App
Academic demonstration of a machine‑learning car pricing model.

### 🔍 Features Used
- Production Year  
- Engine Volume  
- Mileage  
- Leather Interior  
- Cylinders  

### 🧠 Model
Linear Regression  
Optional Standard Scaling

### 👨‍💻 Developer
**Victor Kwabena Opare‑Addo**

### 📦 Version
Academic Edition – Clean UI
""")

# --------------------------------------------------------
# MAIN HEADER
# --------------------------------------------------------
st.markdown('<div class="title">🚗 Car Price Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">A clean, modern & fully responsive academic interface</div>', unsafe_allow_html=True)

# --------------------------------------------------------
# LOAD MODEL / SCALER
# --------------------------------------------------------
MODEL_PATH = "car_price_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file missing! Upload 'car_price_model.pkl'.")
    st.stop()

model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb")) if os.path.exists(SCALER_PATH) else None

# --------------------------------------------------------
# TOP PREDICTION BOX (starts empty)
# --------------------------------------------------------
prediction_box = st.markdown(
    '<div class="top-card">Enter car details below</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------------
# INPUT FORM
# --------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    prod_year = st.number_input("Production Year", min_value=1900, max_value=2026, value=2015)
    engine_volume = st.number_input("Engine Volume (Liters)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)

with col2:
    leather = st.selectbox("Leather Interior", ["No", "Yes"])
    mileage = st.number_input("Mileage (KM)", min_value=0, max_value=1_000_000, value=50000)

cyl = st.number_input("Number of Cylinders", min_value=1, max_value=16, value=4)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------
# PROCESS INPUT
# --------------------------------------------------------
leather_val = 1 if leather == "Yes" else 0
data = np.array([[prod_year, leather_val, engine_volume, mileage, cyl]])

if scaler:
    data = scaler.transform(data)

# --------------------------------------------------------
# PREDICT BUTTON
# --------------------------------------------------------
if st.button("Predict Price"):
    price = float(model.predict(data)[0])

    prediction_box.markdown(
        f'<div class="top-card">Estimated Price: GHS {price:,.2f}</div>',
        unsafe_allow_html=True
    )

# --------------------------------------------------------
# FOOTER
# --------------------------------------------------------
st.markdown("""
<hr>
<div style='text-align:center; color:white; text-shadow:2px 2px 6px black;'>
Academic Project • Streamlit Interface • <b>Victor Opare‑Addo</b>
</div>
""", unsafe_allow_html=True)
