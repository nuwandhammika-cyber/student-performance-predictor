import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

st.title("Online Shopper Intention Predictor")
st.write("Enter session details below to predict if the visitor will generate revenue.")

# Create input fields for all features (Simplified for demonstration)
# Note: In a real app, you would add inputs for all 26 columns used in training
admin = st.number_input("Administrative", min_value=0)
admin_dur = st.number_input("Administrative Duration", min_value=0.0)
prod_rel = st.number_input("Product Related", min_value=0)
prod_rel_dur = st.number_input("Product Related Duration", min_value=0.0)
bounce = st.slider("Bounce Rates", 0.0, 1.0, 0.01)
exit = st.slider("Exit Rates", 0.0, 1.0, 0.01)
page_val = st.number_input("Page Values", min_value=0.0)

# Placeholder for the rest of the features (using mean/default values for simplicity)
# Ensure the input dataframe matches the structure of X_train
if st.button("Predict"):
    # Create a dummy row with the same columns as the trained model
    # You should map your inputs to the correct column names from your X_train
    input_data = pd.DataFrame([[admin, admin_dur, 0, 0, prod_rel, prod_rel_dur, bounce, exit, page_val, 0, 1, 2, 1, 1, 
                                False, False, False, False, False, False, False, False, False, False, False, False]],
                              columns=model.feature_names_in_)
    
    prediction = model.predict(input_data)
    
    if prediction[0]:
        st.success("The visitor is likely to generate Revenue!")
    else:
        st.error("The visitor is unlikely to generate Revenue.")
