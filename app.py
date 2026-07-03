import streamlit as st
import pandas as pd
import joblib
# Explicitly import sklearn components to help with unpickling
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load the saved pipeline
# Note: This file must match the one generated in cell DHCnTCMU6quU
try:
    model = joblib.load("model.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")

st.title("Online Shopper Prediction")
st.write("Please enter the session details below:")

# Define inputs
Administrative = st.number_input("Administrative", value=0)
Administrative_Duration = st.number_input("Administrative Duration", value=0.0)
Informational = st.number_input("Informational", value=0)
Informational_Duration = st.number_input("Informational Duration", value=0.0)
ProductRelated = st.number_input("Product Related", value=0)
ProductRelated_Duration = st.number_input("Product Related Duration", value=0.0)
BounceRates = st.number_input("Bounce Rates", value=0.0)
ExitRates = st.number_input("Exit Rates", value=0.0)
PageValues = st.number_input("Page Values", value=0.0)
SpecialDay = st.number_input("Special Day", value=0.0)

Month = st.selectbox("Month", ["Jan","Feb","Mar","Apr","May","June","Jul","Aug","Sep","Oct","Nov","Dec"])
OperatingSystems = st.number_input("Operating Systems", value=1)
Browser = st.number_input("Browser", value=1)
Region = st.number_input("Region", value=1)
TrafficType = st.number_input("Traffic Type", value=1)
VisitorType = st.selectbox("Visitor Type", ["Returning_Visitor", "New_Visitor", "Other"])
Weekend = st.selectbox("Weekend", [True, False])

# Create DataFrame
input_data = pd.DataFrame([{
    "Administrative": Administrative,
    "Administrative_Duration": Administrative_Duration,
    "Informational": Informational,
    "Informational_Duration": Informational_Duration,
    "ProductRelated": ProductRelated,
    "ProductRelated_Duration": ProductRelated_Duration,
    "BounceRates": BounceRates,
    "ExitRates": ExitRates,
    "PageValues": PageValues,
    "SpecialDay": SpecialDay,
    "Month": Month,
    "OperatingSystems": OperatingSystems,
    "Browser": Browser,
    "Region": Region,
    "TrafficType": TrafficType,
    "VisitorType": VisitorType,
    "Weekend": Weekend
}])

if st.button("Predict"):
    try:
        pred = model.predict(input_data)
        if pred[0]:
            st.success("Prediction: The visitor will likely purchase!")
        else:
            st.info("Prediction: The visitor is unlikely to purchase.")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
