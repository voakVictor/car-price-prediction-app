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
# RESPONSIVE + SPACING FIX (CRITICAL FIX INCLUDED)
# --------------------------------------------------------
def apply_styles():
    st.markdown("""
    <style>

    /* --- Fix Streamlit Header Overlap --- */
    .block-container {
        max-width: 720px;
        padding-top: 5rem !important;  /* 🔥 FIXED HERE */
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* --- Background --- */
    .stApp {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: #f1f5f9;
    }

    /* --- Title --- */
    .title {
        font-size: 32px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
        color: #f8fafc;
    }

    /* --- Subtitle --- */
    .subtitle {
        text-align: center;
        font-size: 15px;
        color: #cbd5f5;
        margin-bottom: 25px;
    }

    /* --- Card --- */
    .card {
        background: #1e293b;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #334155;
    }

    /* --- Prediction Box --- */
    .prediction {
        background: #0d6efd;
        padding: 14px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: white;
        margin-bottom: 20px;
    }

    /* --- Labels --- */
    label {
        font-weight: 600 !important;
        color: #e2e8f0 !important;
    }

    /* --- Inputs --- */
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] {
        background-color: #0f172a !important;
        color: white !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
    }

    /* --- Button --- */
    .stButton button {
        width: 100%;
        border-radius: 8px;
        height: 45px;
        font-weight: 600;
        font-size: 16px;
    }

    /* --- AUTO RESPONSIVE --- */
    @media (max-width: 768px) {
        div[data-testid="column"] {
            width: 100% !important;
            flex: 100% !important;
        }

        .title {
            font-size: 24px !important;
        }

        .subtitle {
            font-size: 16px !important;
        }

        .prediction {
            font-size: 17px !important;
        }

        .block-container {
            padding-top: 4rem !important; /* adjust for smaller screens */
        }
    }

    </style>
    """, unsafe_allow_html=True)

apply_styles()

# --------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------
st.sidebar.title("Application Information")
st.sidebar.write("""
This application demonstrates the integration of a machine learning model 
into a web interface using Streamlit.

**Model:** Linear Regression  
**Features Used:**
- Production Year  
- Engine Volume  
- Mileage  
- Leather Interior  
- Cylinders  

**Institution:** GCTU  
""")

# --------------------------------------------------------
# HEADER
# --------------------------------------------------------
st.markdown('<div class="title">Car Price Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning Model Integrated into a Web Application</div>', unsafe_allow_html=True)

# --------------------------------------------------------
# LOAD MODEL
# --------------------------------------------------------
MODEL_PATH = "car_price_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model file missing.")
    st.stop()

model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb")) if os.path.exists(SCALER_PATH) else None

# --------------------------------------------------------
# PREDICTION DISPLAY
# --------------------------------------------------------
prediction_placeholder = st.empty()
prediction_placeholder.markdown(
    '<div class="prediction">Enter values and click Predict</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------------
# INPUT FORM
# --------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")

with col1:
    prod_year = st.number_input("Production Year", 1900, 2026, 2015)
    engine_volume = st.number_input("Engine Volume (L)", 0.5, 10.0, 2.0)

with col2:
    leather = st.selectbox("Leather Interior", ["No", "Yes"])
    mileage = st.number_input("Mileage (KM)", 0, 1_000_000, 50000)

cyl = st.number_input("Number of Cylinders", 1, 16, 4)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------
# DATA PROCESSING
# --------------------------------------------------------
leather_val = 1 if leather == "Yes" else 0

input_data = np.array([[
    prod_year,
    leather_val,
    engine_volume,
    mileage,
    cyl
]])

if scaler:
    input_data = scaler.transform(input_data)

# --------------------------------------------------------
# PREDICTION
# --------------------------------------------------------
if st.button("Predict Price"):
    try:
        prediction = float(model.predict(input_data)[0])

        prediction_placeholder.markdown(
            f'<div class="prediction">Estimated Price: GHS {prediction:,.2f}</div>',
            unsafe_allow_html=True
        )

    except Exception:
        st.error("Prediction failed.")

# FOOTER

st.markdown("""
<hr>
<center style='color:#94a3b8'>
Victor Kwabena Opare-Addo  •  GCTU  •  @2026
</center>
""", unsafe_allow_html=True)
