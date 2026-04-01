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
# RESPONSIVE DARK BACKGROUND IMAGE
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
            max-width: 750px !important;
        }

        @media (max-width: 640px) {
            .stNumberInput, .stSelectbox {
                width: 100% !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

# --------------------------------------------------------
# CLEAN UI CSS
# --------------------------------------------------------
st.markdown("""
    <style>

        .title {
            font-size: 34px;
            font-weight: 900;
            color: white;
            text-align: center;
            text-shadow: 3px 3px 10px black;
            margin-bottom: 0.3rem;
        }

        .subtitle {
            font-size: 16px;
            color: #efefef;
            text-align: center;
            margin-bottom: 1.2rem;
            text-shadow: 2px 2px 6px black;
        }

        .top-card {
            background: rgba(13, 110, 253, 0.92);
            padding: 15px;
            border-radius: 12px;
            color: white;
            font-size: 20px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 1rem;
            box-shadow: 0 6px 25px rgba(0, 0, 0, 0.85);
        }

        .card {
            background: rgba(0, 0, 0, 0.55);
            padding: 18px;
            border-radius: 14px;
            box-shadow: 0 6px 25px rgba(0,0,0,0.9);
            backdrop-filter: blur(8px);
        }

        label {
            color: white !important;
            font-weight: 600 !important;
            text-shadow: 2px 2px 6px black;
        }

        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] {
            background-color: rgba(255, 255, 255, 0.18) !important;
            color: white !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255,255,255,0.4);
        }

        .stSelectbox div[role="listbox"] {
            background-color: rgba(15,15,15,0.95) !important;
            color: white !important;
        }

    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# SIDEBAR INFO PANEL
# --------------------------------------------------------
st.sidebar.title("ℹ️ App Information")

st.sidebar.markdown("""
### 🚗 Car Price Prediction App
This application uses a **machine learning model** to estimate the price of a car based on:

- Production Year  
- Leather Interior  
- Engine Volume  
- Mileage  
- Number of Cylinders  

### 📦 Model Details
- Type: Linear Regression  
- Preprocessing: Optional Standard Scaling  
- Input shape: 5 features  

### 🛠 How to Use
1. Enter the details of the car  
2. Click **Predict Price**  
3. View the result instantly at the top  

### 👨‍💻 Developer
**Victor Kwabena Opare‑Addo**

### 🔗 Version
`v1.0 — Responsive UI Update`

""")

# --------------------------------------------------------
# MAIN HEADER
# --------------------------------------------------------
st.markdown('<div class="title">🚗 Car Price Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Get an instant estimate for your vehicle</div>', unsafe_allow_html=True)

# --------------------------------------------------------
# LOAD MODEL & SCALER
# --------------------------------------------------------
MODEL_PATH = "car_price_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ 'car_price_model.pkl' not found.")
    st.stop()

model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb")) if os.path.exists(SCALER_PATH) else None

# --------------------------------------------------------
# PREDICTION DISPLAY BOX
# --------------------------------------------------------
prediction_box = st.markdown(
    '<div class="top-card">Enter your details below</div>',
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
    leather_interior = st.selectbox("Leather Interior", ["No", "Yes"])
    mileage = st.number_input("Mileage (KM)", 0, 1_000_000, 50000)

cylinders = st.number_input("Number of Cylinders", 1, 16, 4)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------
# PROCESS INPUT
# --------------------------------------------------------
leather_interior = 1 if leather_interior == "Yes" else 0
data = np.array([[prod_year, leather_interior, engine_volume, mileage, cylinders]])

if scaler is not None:
    data = scaler.transform(data)

# --------------------------------------------------------
# PREDICT BUTTON
# --------------------------------------------------------
if st.button("Predict Price ☑️"):
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
    <div style='text-align:center;color:white;text-shadow:2px 2px 6px black;'>
        Built by <strong>Victor Kwabena Opare‑Addo</strong> • Powered by Streamlit 🚀
    </div>
""", unsafe_allow_html=True)
