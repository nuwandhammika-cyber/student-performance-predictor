import streamlit as st
import pandas as pd
import joblib
import sklearn

st.set_page_config(page_title="Shopper Intention Predictor")
st.title("Online Shopper Prediction")
st.write(f"Running with scikit-learn version: {sklearn.__version__}")

@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl")
    except Exception as e:
        st.error(f"Version Mismatch Error: {e}")
        st.info("Ensure your GitHub requirements.txt matches the Colab version (1.6.1)")
        return None

model = load_model()

if model:
    st.write("### Enter Session Details")
    # Use columns for a cleaner layout
    col1, col2 = st.columns(2)
    with col1:
        adm = st.number_input("Administrative", 0)
        inf = st.number_input("Informational", 0)
        pro = st.number_input("Product Related", 0)
        br = st.number_input("Bounce Rates", 0.0, format="%.4f")
        pv = st.number_input("Page Values", 0.0)
    with col2:
        adm_d = st.number_input("Admin Duration", 0.0)
        inf_d = st.number_input("Info Duration", 0.0)
        pro_d = st.number_input("Product Duration", 0.0)
        er = st.number_input("Exit Rates", 0.0, format="%.4f")
        sd = st.number_input("Special Day", 0.0)

    month = st.selectbox("Month", ["Jan","Feb","Mar","Apr","May","June","Jul","Aug","Sep","Oct","Nov","Dec"])
    os_sys = st.number_input("Operating Systems", 1)
    browser = st.number_input("Browser", 1)
    region = st.number_input("Region", 1)
    traffic = st.number_input("Traffic Type", 1)
    visitor = st.selectbox("Visitor Type", ["Returning_Visitor", "New_Visitor", "Other"])
    weekend = st.selectbox("Weekend", [True, False])

    input_data = pd.DataFrame([{
        "Administrative": adm, "Administrative_Duration": adm_d,
        "Informational": inf, "Informational_Duration": inf_d,
        "ProductRelated": pro, "ProductRelated_Duration": pro_d,
        "BounceRates": br, "ExitRates": er, "PageValues": pv,
        "SpecialDay": sd, "Month": month, "OperatingSystems": os_sys,
        "Browser": browser, "Region": region, "TrafficType": traffic,
        "VisitorType": visitor, "Weekend": weekend
    }])

    if st.button("Predict"):
        prediction = model.predict(input_data)
        if prediction[0]:
            st.success("Outcome: Revenue Likely!")
        else:
            st.warning("Outcome: No Revenue Likely.")
