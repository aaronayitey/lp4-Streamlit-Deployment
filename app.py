import streamlit as st
import joblib
import numpy as np
import pandas as pd
# Machine Learning Modeling
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error



# Load the pre-trained model and preprocessor
model = joblib.load('./xgb_model.joblib')
preprocessor = joblib.load('./preprocessor.joblib')

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
    # Create a date input using st.date_input
    date = st.date_input("Enter Date")  

    # Convert the selected date to a string in the desired format (e.g., YYYY-MM-DD)
    formatted_date = date.strftime('%Y-%m-%d')  

# User Inputs - Year
with col2:
    family = st.selectbox("Select product family", ['CELEBRATION', 'CLEANING', 'DAIRY', 'DELI', 'EGGS', 'FROZEN FOODS',
       'GROCERY I', 'GROCERY II', 'HARDWARE', 'HOME AND KITCHEN I',
       'HOME AND KITCHEN II', 'HOME APPLIANCES', 'HOME CARE',
       'LADIESWEAR', 'LAWN AND GARDEN', 'LINGERIE', 'LIQUOR,WINE,BEER',
       'MAGAZINES', 'MEATS', 'PERSONAL CARE', 'PET SUPPLIES',
       'PLAYERS AND ELECTRONICS', 'POULTRY', 'PREPARED FOODS', 'PRODUCE',
       'SCHOOL AND OFFICE SUPPLIES', 'SEAFOOD', 'AUTOMOTIVE', 'BABY CARE',
       'BEAUTY', 'BEVERAGES', 'BOOKS', 'BREAD/BAKERY'])

# User Inputs - On Promotion
with col1:
    onpromotion = st.number_input("Enter Number for onpromotion", min_value=0, step=1)

# # Add custom spacing between columns
# st.markdown("<hr>", unsafe_allow_html=True)

# User Inputs - Day of the Week
with col2:
    city = st.selectbox("Select city", ['Quito', 'Cayambe', 'Latacunga', 'Riobamba', 'Ibarra',
       'Santo Domingo', 'Guaranda', 'Puyo', 'Ambato', 'Guayaquil',
       'Salinas', 'Daule', 'Babahoyo', 'Quevedo', 'Playas', 'Libertad',
       'Cuenca', 'Loja', 'Machala', 'Esmeraldas', 'Manta', 'El Carmen'])

# User Inputs - Product Category
with col1:
    oil_prices = st.number_input("Enter Number for oil prices", min_value=1, step=1)


# # Add custom spacing between columns
# st.markdown("<hr>", unsafe_allow_html=True)

# User Inputs - Day of the Week
with col2:
    holiday_type = st.selectbox("Select holiday type", ['Holiday', 'Additional', 'Transfer', 'Event', 'Bridge'])

# User Inputs - Product Category
with col1:
    sales_lag_1 = st.number_input("Enter Number for sales lag", min_value=0, step=1)


# User Inputs - Day of the Week
with col2:
    moving_average = st.number_input("Enter Number for moving average", min_value=0, step=1)

# Placeholder for Predicted Value

# Add custom spacing between columns
st.markdown("<hr>", unsafe_allow_html=True)


# Predict button with custom CSS style
predict_button_style = (
    f"background-color: #3498db; color: white; border-radius: 5px; padding: 10px; cursor: pointer;"
)
button_clicked = st.button("Predict", key="predict_button", on_click=None)
if button_clicked:
    # Prepare input data for prediction
    # Prepare input data for prediction
    # Create a DataFrame with all required columns except "sales"
    prediction_placeholder = st.empty()
    input_df = pd.DataFrame({
        "family": [family],
        "onpromotion": [onpromotion],
        "city": [city],
        "oil_prices": [oil_prices],
        "holiday_type": [holiday_type],
        "sales_lag_1": [sales_lag_1],
        "moving_average": [moving_average]
    })
    
    # Transform the input DataFrame using the preprocessor
    preprocessed_data = preprocessor.transform(input_df)
    


    # Make a prediction
    prediction = model.predict(preprocessed_data)

    # Display the prediction
    prediction_placeholder.text(f"Predicted Value for sales: {prediction[0]}")
   
    if prediction >= 0:
        prediction_placeholder.markdown(
            f'Predicted Value for sales: <span style="background-color: green; padding: 2px 5px; border-radius: 5px;">{prediction[0]}</span>',
            unsafe_allow_html=True
        )
    else:
        prediction_placeholder.markdown(
            f'Predicted Value for sales: <span style="background-color: red; padding: 2px 5px; border-radius: 5px;">{prediction[0]}</span>',
            unsafe_allow_html=True
        )

   
