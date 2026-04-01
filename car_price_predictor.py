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
# BACKGROUND IMAGE (safe, direct URL)
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
            padding-top: 1.2rem !important;
            max-width: 760px !important;
        }

        .title {
            font-size: 32px;
            font-weight: 900;
            color: white;
            text-align: center;
            text-shadow: 3px 3px 8px black;
        }

        .subtitle {
            font-size: 15px;
            color: #ececec;
            text-align: center;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 5px black;
        }

        .top-card {
            background: rgba(13,110,253,0.88);
            color: white;
            padding: 12px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 1rem;
        }

        .card {
            background: rgba(0,0,0,0.55);
            padding: 18px;
            border-radius: 14px;
        }

        label {
            color: white !important;
            font-weight: 600 !important;
            text-shadow: 2px 2px 5px black;
        }

        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] {
            background-color: rgba(255,255,255,0.22) !important;
            color: white !important;
            border-radius: 8px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

# --------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------
st.sidebar.title("ℹ️ App Information")
st.sidebar.markdown("""
### Car Price Prediction System  
This system uses a trained **machine learning model** to estimate car prices based on:

- Production Year  
- Engine Volume  
- Mileage  
- Leather Interior  
- Number of Cylinders  

### Purpose  
Built for academic demonstration and learning Streamlit.

### Developer  
**Victor Kwabena Opare‑Addo**

### Version  
1.0 (Clean Academic Edition)
""")

# --------------------------------------------------------
# PAGE HEADER
# --------------------------------------------------------
st.markdown('<div class="title">🚗 Car Price Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Simple and clean interface for academic demonstration</div>', unsafe_allow_html=True)

# --------------------------------------------------------
# LOAD MODEL & SCALER
# --------------------------------------------------------
MODEL_PATH = "car_price_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file missing! Upload 'car_price_model.pkl'.")
    st.stop()

model = pickle.load(open(MODEL_PATH, "rb"))

scaler = None
if os.path.exists(SCALER_PATH):
    scaler = pickle.load(open(SCALER_PATH, "rb"))

# --------------------------------------------------------
# TOP PREDICTION BOX (EMPTY AT START)
# --------------------------------------------------------
prediction_box = st.markdown(
    '<div class="top-card">Enter details below</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------------
# INPUT FORM (SIMPLE & CLEAN)
# --------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Production Year", 1900, 2026, 2015)
    engine = st.number_input("Engine Volume (L)", 0.5, 10.0, 2.0, step=0.1)

with col2:
    leather = st.selectbox("Leather Interior", ["No", "Yes"])
    mileage = st.number_input("Mileage (KM)", 0, 1_000_000, 50000)

cyl = st.number_input("Cylinders", 1, 16, 4)

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------
# PROCESS DATA
# --------------------------------------------------------
leather_num = 1 if leather == "Yes" else 0

input_data = np.array([[year, leather_num, engine, mileage, cyl]])

if scaler:
    input_data = scaler.transform(input_data)

# --------------------------------------------------------
# PREDICT BUTTON
# --------------------------------------------------------
if st.button("Predict Price"):
    price = float(model.predict(input_data)[0])

    prediction_box.markdown(
        f'<div class="top-card">Estimated Price: GHS {price:,.2f}</div>',
        unsafe_allow_html=True
    )

# --------------------------------------------------------
# FOOTER
# --------------------------------------------------------
st.markdown("""
<hr style="border: 1px solid white;">
<div style='text-align:center;color:white;text-shadow:2px 2px 6px black;'>
Academic Assignment • Built with Streamlit • By <b>Victor Opare‑Addo</b>
</div>
""", unsafe_allow_html=True)
