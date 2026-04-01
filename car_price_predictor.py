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
# RESPONSIVE STYLING (IMPROVED)
# --------------------------------------------------------
def apply_styles():
    st.markdown("""
    <style>

    /* --- Background --- */
    .stApp {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: #f1f5f9;
    }

    /* --- Container --- */
    .block-container {
        max-width: 720px;
        padding: 1rem;
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
        margin-bottom: 20px;
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

    /* --- RESPONSIVE DESIGN --- */

    /* Tablets */
    @media (max-width: 900px) {
        .block-container {
            max-width: 90% !important;
        }
    }

    /* Mobile */
    @media (max-width: 600px) {
        .title {
            font-size: 24px !important;
        }

        .subtitle {
            font-size: 14px !important;
        }

        .prediction {
            font-size: 17px !important;
            padding: 12px;
        }

        .card {
            padding: 15px !important;
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

**Purpose:** Academic Assignment  
""")

# --------------------------------------------------------
# HEADER
# --------------------------------------------------------
st.markdown('<div class="title">🚗 Car Price Prediction System</div>', unsafe_allow_html=True)
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
# INPUT FORM (BETTER RESPONSIVE LOGIC)
# --------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

# Use responsive columns
is_mobile = st.checkbox("📱 Use mobile layout", value=False)

if is_mobile:
    # STACKED layout (better for small screens)
    prod_year = st.number_input("Production Year", 1900, 2026, 2015)
    engine_volume = st.number_input("Engine Volume (L)", 0.5, 10.0, 2.0)
    leather = st.selectbox("Leather Interior", ["No", "Yes"])
    mileage = st.number_input("Mileage (KM)", 0, 1_000_000, 50000)
else:
    col1, col2 = st.columns(2)
    with col1:
        prod_year = st.number_input("Production Year", 1900, 2026, 2015)
        engine_volume = st.number_input("Engine Volume (L)", 0.5, 10.0, 2.0)
    with col2:
        leather = st.selectbox("Leather Interior", ["No", "Yes"])
        mileage = st.number_input("Mileage (KM)", 0, 1_000_000, 50000)

# Full width
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

# --------------------------------------------------------
# FOOTER
# --------------------------------------------------------
st.markdown("""
<hr>
<center style='color:#94a3b8'>
Academic Project • Streamlit • Victor Opare-Addo
</center>
""", unsafe_allow_html=True)
