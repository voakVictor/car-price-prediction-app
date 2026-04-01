import streamlit as st
import numpy as np
import pickle
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Car Price Prediction System",
    page_icon="🚗",
    layout="centered"
)

# -----------------------------
# Load Model and Scaler
# -----------------------------
MODEL_PATH = "car_price_model.pkl"
SCALER_PATH = "scaler.pkl"  # Ensure you saved this during training

# Check if files exist
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please ensure 'car_price_model.pkl' exists.")
    st.stop()

model = pickle.load(open(MODEL_PATH, "rb"))

# Load scaler if available
scaler = None
if os.path.exists(SCALER_PATH):
    scaler = pickle.load(open(SCALER_PATH, "rb"))
else:
    st.warning("Scaler not found. Predictions may be inaccurate if model requires scaling.")

# -----------------------------
# UI Design
# -----------------------------
st.title("🚗 Car Price Prediction System")

st.write(
    "This intelligent system estimates the price of a car using a trained Linear Regression machine learning model. "
    "Kindly provide the required details below to obtain a prediction instantly, without the need for any manual calculations. "
)

st.header("Please Enter  Your Car Details")

# -----------------------------
# Input Section
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    prod_year = st.number_input(
        "Production Year",
        min_value=1900,
        max_value=2026,
        value=2015,
        help="The year the car was manufactured."
    )

with col2:
    leather_interior = st.selectbox(
        "Leather Interior",
        ["No", "Yes"],
        help="Does the car have leather interior? This can affect the price."
    )

engine_volume = st.number_input(
    "Engine Volume (Liters)",
    min_value=0.5,
    max_value=10.0,
    value=2.0,
    step=0.1,
    help="The volume of the car's engine in liters. Larger engines typically cost more."
)

mileage = st.number_input(
    "Mileage (KM)",
    min_value=0,
    max_value=1000000,
    value=50000,
    help="The total distance the car has traveled in kilometers. Higher mileage usually reduces the price."
)

cylinders = st.number_input(
    "Number of Cylinders",
    min_value=1,
    max_value=16,
    value=4,
    help="The number of cylinders in the car's engine. More cylinders can indicate a more powerful (and expensive) car."
)

# -----------------------------
# Data Processing
# -----------------------------
# Convert categorical to numerical
leather_interior = 1 if leather_interior == "Yes" else 0

# Create input array (IMPORTANT: Order must match training)
input_data = np.array([[
    prod_year,
    leather_interior,
    engine_volume,
    mileage,
    cylinders
]])

# Apply scaling if scaler exists
if scaler is not None:
    try:
        input_data = scaler.transform(input_data)
    except Exception as e:
        st.error(f"Error during scaling: {e}")
        st.stop()

# -----------------------------
# Prediction Section
# -----------------------------
st.header("Prediction")

if st.button("Predict Price ☑️"):
    try:
        prediction = model.predict(input_data)

        # Format output nicely
        predicted_price = float(prediction[0])

        st.success(f"Estimated Car Price: GHS{predicted_price:,.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")