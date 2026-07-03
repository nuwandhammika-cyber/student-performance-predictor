import streamlit as st
import pandas as pd
import joblib

# Load full pipeline
model = joblib.load("model.pkl")

st.set_page_config(page_title="Online Shopper Predictor", layout="centered")

st.title("🛒 Online Shopper Intention Predictor")

st.write("Enter session details to predict purchase behavior")

st.markdown("---")

# ======================
# INPUTS (MATCH DATASET)
# ======================

Administrative = st.number_input("Administrative", 0)
Administrative_Duration = st.number_input("Administrative Duration", 0.0)

Informational = st.number_input("Informational", 0)
Informational_Duration = st.number_input("Informational Duration", 0.0)

ProductRelated = st.number_input("Product Related", 0)
ProductRelated_Duration = st.number_input("Product Related Duration", 0.0)

BounceRates = st.number_input("Bounce Rates", 0.0)
ExitRates = st.number_input("Exit Rates", 0.0)
PageValues = st.number_input("Page Values", 0.0)

SpecialDay = st.number_input("Special Day", 0.0)

Month = st.selectbox("Month",
    ["Jan","Feb","Mar","Apr","May","June","Jul","Aug","Sep","Oct","Nov","Dec"]
)

OperatingSystems = st.number_input("Operating Systems", 1)
Browser = st.number_input("Browser", 1)
Region = st.number_input("Region", 1)
TrafficType = st.number_input("Traffic Type", 1)

VisitorType = st.selectbox("Visitor Type",
    ["Returning_Visitor", "New_Visitor", "Other"]
)

Weekend = st.selectbox("Weekend", ["True", "False"])

# ======================
# CREATE DATAFRAME (SAFE)
# ======================

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

# ======================
# PREDICTION
# ======================

if st.button("Predict"):

    prediction = model.predict(input_data)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)[0][1]
        st.write(f"Probability of Purchase: {prob:.2f}")

    if prediction[0]:
        st.success("Customer WILL purchase")
    else:
        st.error("Customer will NOT purchase")
