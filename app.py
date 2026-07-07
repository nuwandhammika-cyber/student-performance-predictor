import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file 'model.pkl' not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

st.set_page_config(page_title='Customer Purchase Predictor', layout='centered')

# Define the exact columns that the model expects after one-hot encoding
# These must match the X.columns from the training phase
MODEL_FEATURE_COLUMNS = ['Age', 'AnnualIncome', 'NumberOfPurchases', 'TimeSpentOnWebsite',
                         'Gender_1',
                         'ProductCategory_1', 'ProductCategory_2', 'ProductCategory_3', 'ProductCategory_4',
                         'LoyaltyProgram_1',
                         'DiscountsAvailed_1', 'DiscountsAvailed_2', 'DiscountsAvailed_3', 'DiscountsAvailed_4', 'DiscountsAvailed_5']

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
    income = st.number_input('Annual Income (LKR)', min_value=10000, max_value=500000, value=50000, step=1000)
    purchases = st.number_input('Number of Previous Purchases', min_value=0, value=5)

with col_b:
    category_labels = {"Electronics": 0, "Clothing": 1, "Home Goods": 2, "Beauty": 3, "Sports": 4}
    selected_category_name = st.selectbox("Product Category", options=list(category_labels.keys()))
    category_id = category_labels[selected_category_name]
    time_spent = st.number_input('Time Spent on Website (minutes)', min_value=0.0, max_value=180.0, value=20.0)
    loyalty = st.selectbox('Loyalty Program Member', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    discounts = st.slider('Discounts Availed (0-5)', 0, 5, 0)

# Predict Button
if st.button('Predict Purchase Status'):
    # Prepare input data as a DataFrame, mimicking the training data structure
    input_data_dict = {
        'Age': [age],
        'AnnualIncome': [income],
        'NumberOfPurchases': [purchases],
        'TimeSpentOnWebsite': [time_spent],
        'Gender': [gender_val],
        'ProductCategory': [category_id],
        'LoyaltyProgram': [loyalty],
        'DiscountsAvailed': [discounts]
    }
    input_df = pd.DataFrame(input_data_dict)

    # Apply one-hot encoding consistent with training data (drop_first=True)
    input_df_encoded = pd.get_dummies(input_df, columns=['Gender', 'ProductCategory', 'LoyaltyProgram', 'DiscountsAvailed'], drop_first=True)

    # Ensure all model feature columns are present and in the correct order, fill missing with 0
    final_input_features = pd.DataFrame(columns=MODEL_FEATURE_COLUMNS)
    final_input_features = pd.concat([final_input_features, input_df_encoded], ignore_index=True)
    final_input_features = final_input_features.fillna(0) # Fill NaN for missing dummy columns

    # Convert boolean columns to int for compatibility with the model
    for col in final_input_features.columns:
        if final_input_features[col].dtype == 'bool':
            final_input_features[col] = final_input_features[col].astype(int)

    prediction = model.predict(final_input_features)
    probability = model.predict_proba(final_input_features)[0][1]

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
    st.title(" 📘 Project Information")
    st.markdown("────────────────────")
    st.markdown("**👨‍🎓 Student Details**")
    st.write("Name:\nP. Nuwan Dhammika Perera")
    st.write("Student ID:\nS25021946")
    st.markdown("────────────────────")
    st.markdown("**🏫 University**")
    st.write("Wrexham University")
    st.markdown("────────────────────")
    st.markdown("**📚 Module**")
    st.write("COM763\nAdvanced Machine Learning")
    st.markdown("────────────────────")
    st.markdown("**📈 Project**")
    st.write("Customer Purchase Prediction")
    st.write("**Model:**\nRandom Forest Classifier")
    st.write("**Accuracy:**\n91.01%")
    st.markdown("────────────────────")
