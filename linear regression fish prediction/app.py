import streamlit as st
import pickle
import numpy as np

with open("Fish_weight_prediction.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Fish Weight Prediction App")

length3 = st.number_input("Enter Height", min_value=0.0, step=0.1)

if st.button("Predict Weight"):
    input_data = np.array([[length3]])
    prediction = model.predict(input_data)

    predicted_weight = prediction.item()  #  important line
    st.success(f" Predicted Fish Weight: {predicted_weight:.2f} grams")
