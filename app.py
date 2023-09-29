import streamlit as st
import joblib
import numpy as np

# Load the pre-trained model and preprocessor
# model = joblib.load('model.pkl')
# preprocessor = joblib.load('preprocessor.pkl')

# Set the page layout to full width
st.set_page_config(layout="wide")

# Custom CSS styling for the title
st.markdown(
    """
    <style>
    .title-text {
        font-size: 28px;
        text-align: center;
        background-color: #3498db;
        color: white;
        padding: 10px 0;
        width: 100%;
        position: sticky;
        top: 0;
        z-index: 1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit App Title
st.markdown('<p class="title-text">Machine Learning App for Sales Prediction</p>', unsafe_allow_html=True)

# Create a layout with 2 columns for even distribution
col1, col2 = st.columns(2)

# User Inputs - Number
with col1:
    number = st.number_input("Enter Number", min_value=1, step=1)

# User Inputs - Year
with col2:
    year = st.slider("Select Year", 2013, 2023, 2023)

# Add custom spacing between columns
st.markdown("<hr>", unsafe_allow_html=True)

# User Inputs - Transaction
with col1:
    transaction = st.number_input("Enter Transaction", min_value=1, step=1)

# User Inputs - On Promotion
with col2:
    onpromotion = st.selectbox("Select On Promotion", ["Yes", "No"])

# Add custom spacing between columns
st.markdown("<hr>", unsafe_allow_html=True)

# User Inputs - Day of the Week
with col1:
    day_of_week = st.selectbox("Select Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

# User Inputs - Product Category
with col2:
    product_category = st.text_input("Enter Product Category")

# Placeholder for Predicted Value
prediction_placeholder = st.empty()

# Predict Button
if st.button("Predict"):
    # Prepare input data for prediction
    input_data = np.array([[number, year, transaction, onpromotion, day_of_week, product_category]])
    # preprocessed_data = preprocessor.transform(input_data)

    # Make a prediction
    # prediction = model.predict(preprocessed_data)

    # Display the prediction
    # prediction_placeholder.text(f"Predicted Value: {prediction[0]}")
