# ===============================
# app.py
# ===============================
import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
mlp = joblib.load("mlp_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="NSFWQI Prediction", layout="centered")

st.title("🌊 NSFWQI Prediction System")
st.write("Enter water quality parameters to calculate the NSFWQI:")

# Input fields
NO3 = st.number_input("NO3 (mg/L)", min_value=0.0, step=0.01)
COD = st.number_input("COD (mg/L)", min_value=0.0, step=0.01)
FC = st.number_input("Fecal Coliform (CFU/100ml)", min_value=0.0, step=0.01)
BOD = st.number_input("BOD (mg/L)", min_value=0.0, step=0.01)
TDS = st.number_input("TDS (mg/L)", min_value=0.0, step=0.01)

# Predict button
if st.button("🔎 NSFWQI"):
    features = np.array([[NO3, COD, FC, BOD, TDS]])
    features_scaled = scaler.transform(features)
    prediction = mlp.predict(features_scaled)[0]
    
    st.success(f"✅ NSFWQI: {prediction:.2f}")
