import streamlit as st
import pandas as pd
import joblib


st.set_page_config(page_title="Wine Type Prediction- ML assighmenet - Omar altarakieh", layout="wide")

#  dataset
@st.experimental_memo
def load_data():
    data = pd.read_csv('wine_fraud.csv')
    return data

wine_data = load_data()

#  the trained RandomForest model
model = joblib.load('wine_rf_clf_8-11.pkl')

#  the label encoder
label_encoder = joblib.load('label_encoder.pkl')  # Ensure this is the correct path to your label encoder file

# Define a function to make predictions
def predict_wine_type(features):
    features_df = pd.DataFrame(features, index=[0])
    prediction = model.predict(features_df)
    probability = model.predict_proba(features_df).max()
    return prediction, probability

# Title and introduction to the app
st.title("Wine Type Prediction App")
st.write("Input the wine features to predict the type of wine (red or white).")

# Collecting user inputs for each feature
fixed_acidity = st.number_input('Fixed Acidity', min_value=0.0, format="%.2f")
volatile_acidity = st.number_input('Volatile Acidity', min_value=0.0, format="%.2f")
citric_acid = st.number_input('Citric Acid', min_value=0.0, format="%.2f")
residual_sugar = st.number_input('Residual Sugar', min_value=0.0, format="%.2f")
chlorides = st.number_input('Chlorides', min_value=0.0, format="%.2f")
free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', min_value=0.0, format="%.2f")
total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', min_value=0.0, format="%.2f")
density = st.number_input('Density', min_value=0.0, format="%.4f")
pH = st.number_input('pH', min_value=0.0, format="%.2f")
sulphates = st.number_input('Sulphates', min_value=0.0, format="%.2f")
alcohol = st.number_input('Alcohol', min_value=0.0, format="%.2f")

#  label encoder to provide options
quality_options = label_encoder.classes_
quality = st.selectbox('Quality', options=quality_options)

# Button to make prediction
if st.button('Predict Type of Wine'):
    # Encode the quality input using the label encoder
    encoded_quality = label_encoder.transform([quality])[0]

    features = {
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol,
        'quality': encoded_quality  # Use the encoded quality
    }
    prediction, probability = predict_wine_type(features)
    st.write(f'Predicted Wine Type: {"Red" if prediction[0] == 0 else "White"} with a probability of {probability:.2f}')


