import streamlit as st
import pickle
import numpy as np
import os

# ---------------------------
# Page Config (sab se upar)
# ---------------------------
st.set_page_config(
    page_title="Weight Prediction App",
    layout="centered"
)

# ---------------------------
# Load trained model (Cloud-safe)
# ---------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "weight_prediction.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model

model = load_model()

# ---------------------------
# UI
# ---------------------------
st.title("Weight Prediction Application")
st.write("Predict body weight based on height using a trained machine learning model.")

st.divider()

# ---------------------------
# User Input
# ---------------------------
height_cm = st.number_input(
    label="Enter Height (in centimeters)",
    min_value=100.0,
    max_value=250.0,
    value=170.0,
    step=0.1
)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Weight"):
    try:
        # NOTE:
        # Agar model cm pe train hua hai to ye bilkul theek hai
        input_data = np.array([[height_cm]])
        prediction = model.predict(input_data)

        st.success(f"Predicted Weight: {prediction[0]:.2f} kg")

    except Exception as e:
        st.error("Prediction failed. Check model input format.")
        st.code(str(e))

st.divider()
st.caption("Machine Learning Model by Hassan Bhai")
