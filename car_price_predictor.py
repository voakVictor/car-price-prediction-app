import streamlit as st
import numpy as np
import pickle
import os

# --------------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------------
st.set_page_config(
    page_title="Car Price Prediction System",
    page_icon="🚗",
    layout="centered"
)

# --------------------------------------------------------
# CLEAN PROFESSIONAL STYLING
# --------------------------------------------------------
def apply_styles():
    st.markdown("""
    <style>

    .stApp {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: #f1f5f9;
    }

    .block-container {
        max-width: 700px;
        padding-top: 2rem;
    }

    .title {
        font-size: 32px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
        color: #f8fafc;
    }

    .subtitle {
        text-align: center;
        font-size: 15px;
        color: #cbd5f5;
        margin-bottom: 20px;
    }

    .card {
        background: #1e293b;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #334155;
    }

    .prediction {
        background: #0d6efd;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: white;
        margin-bottom: 20px;
    }

    label {
        font-weight: 600 !important;
        color: #e2e8f0 !important;
    }

    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] {
        background-color: #0f172a !important;
        color: white !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
    }

    .stButton button {
        width: 100%;
        border-radius: 8px;
        height: 45px;
        font-weight: 600;
    }

    @media(max-width: 600px) {
        .title { font-size: 26px; }
        .prediction { font-size: 18px; }
    }

    </style>
    """, unsafe_allow_html=True)

apply_styles()

# --------------------------------------------------------
# SIDEBAR (ACADEMIC INFO)
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
# MODEL EXPLANATION
# --------------------------------------------------------
st.markdown("### 🧠 How the Model Works")
st.info("""
This application uses a Linear Regression model to estimate car prices.

The model learns relationships between car features (year, engine size, mileage, etc.) 
and price based on historical data. When new inputs are provided, it applies learned 
coefficients to predict the car price.
""")

# --------------------------------------------------------
# LOAD MODEL & SCALER
# --------------------------------------------------------
MODEL_PATH = "car_price_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model file is missing.")
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

col1, col2 = st.columns(2)

with col1:
    prod_year = st.number_input("Production Year", 1900, 2026, 2015)
    engine_volume = st.number_input("Engine Volume (L)", 0.5, 10.0, 2.0)

with col2:
    leather = st.selectbox("Leather Interior", ["No", "Yes"])
    mileage = st.number_input("Mileage (KM)", 0, 1_000_000, 50000)

cyl = st.number_input("Number of Cylinders", 1, 16, 4)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------
# INPUT SUMMARY
# --------------------------------------------------------
st.markdown("### 📋 Input Summary")
st.write({
    "Production Year": prod_year,
    "Engine Volume": engine_volume,
    "Mileage": mileage,
    "Cylinders": cyl,
    "Leather Interior": leather
})

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

        # Insight
        if prediction > 500000:
            st.info("This appears to be a high-value vehicle.")
        else:
            st.info("This appears to be a moderately priced vehicle.")

    except Exception as e:
        st.error("Prediction failed.")

# --------------------------------------------------------
# DISCLAIMER
# --------------------------------------------------------
st.warning("""
This prediction is based on a machine learning model trained on historical data. 
Actual market prices may vary due to external economic and market factors.
""")

# --------------------------------------------------------
# FOOTER
# --------------------------------------------------------
st.markdown("""
<hr>
<center style='color:#94a3b8'>
Academic Project • Streamlit • Victor Opare-Addo
</center>
""", unsafe_allow_html=True)
