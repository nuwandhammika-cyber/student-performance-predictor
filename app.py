import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the saved model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'model.pkl' not found.")
    st.stop()

st.set_page_config(page_title='Customer Purchase Predictor', layout='centered')

# Main Title
st.markdown("# 🛒 Customer Purchase Prediction System")
st.divider()

# Project Description Section
st.subheader("Project Description")
st.markdown("""
This application predicts whether a customer is likely to purchase a product based on demographic information and browsing behaviour.
The prediction is generated using a trained **Random Forest Classifier**.
""")
st.divider()

# Customer Information Section
st.subheader("Customer Information")
col_a, col_b = st.columns(2)

with col_a:
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    gender_val = st.selectbox('Gender', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
    income = st.number_input('Annual Income ($)', min_value=10000, max_value=500000, value=50000, step=1000)
    purchases = st.number_input('Number of Previous Purchases', min_value=0, value=5)

with col_b:
    category_labels = ["Electronics", "Clothing", "Home Goods", "Beauty", "Sports"]
    selected_category = st.selectbox("Product Category", options=category_labels)
    category_id = category_labels.index(selected_category)
    time_spent = st.number_input('Time Spent on Website (minutes)', min_value=0.0, max_value=180.0, value=20.0)
    loyalty = st.selectbox('Loyalty Program Member', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    discounts = st.slider('Discounts Availed (0-5)', 0, 5, 0)

# Predict Button
if st.button('Predict Purchase Status'):
    input_data = np.array([[age, gender_val, income, purchases, category_id, time_spent, loyalty, discounts]])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("**Status:** Likely to Purchase")
        st.write(f"**Confidence:** {probability:.2%}")
        st.info("**Recommended Business Action:** Offer personalised promotions or loyalty rewards to finalize the conversion.")
    else:
        st.warning("**Status:** Unlikely to Purchase")
        st.write(f"**Confidence:** {(1-probability):.2%}")
        st.info("**Recommended Business Action:** Implement retargeting ads or high-value discount vouchers to spark interest.")

st.divider()
st.caption("**Disclaimer:** This tool is for educational purposes and provides predictions based on historical patterns.")

# Updated Sidebar
with st.sidebar:
    st.title("📘 Project Information")
    st.markdown("────────────────────────")
    st.markdown("**Student Details**")
    st.write("Name:\nP. Nuwan Dhammika Perera")
    st.write("Student ID:\nS25021946")
    st.markdown("────────────────────────")
    st.markdown("**🏫 University**")
    st.write("Wrexham University")
    st.markdown("────────────────────────")
    st.markdown("**Module**")
    st.write("COM763\nAdvanced Machine Learning")
    st.markdown("────────────────────────")
    st.markdown("**Project**")
    st.write("Customer Purchase Prediction")
    st.write("**Model:**\nRandom Forest Classifier")
    st.write("**Accuracy:**\n94.2%")
    st.markdown("────────────────────────")
