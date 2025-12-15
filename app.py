import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open('heart_disease_model.pkl', 'rb'))

st.title("Heart Disease Predictor System")

st.write("Enter patient details to check heart disease risk")

age = st.number_input("Age", min_value=1)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.number_input("Chest Pain Type (0-3)")
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol Level")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.number_input("Resting ECG (0-2)")
thalach = st.number_input("Max Heart Rate Achieved")
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression")
slope = st.number_input("Slope (0-2)")
ca = st.number_input("Number of Major Vessels (0-3)")
thal = st.number_input("Thal (1-3)")

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                             restecg, thalach, exang, oldpeak,
                             slope, ca, thal]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("No Heart Disease Detected")
