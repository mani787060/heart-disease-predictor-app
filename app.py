import streamlit as st
import numpy as np
import pickle

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction System",
    layout="centered"
)

# -------------------------------
# Load Trained Model
# -------------------------------
model = pickle.load(open("heart_disease_model.pkl", "rb"))

# -------------------------------
# Title & Description
# -------------------------------
st.title("Heart Disease Prediction System")

st.write(
    "This application predicts the likelihood of heart disease based on "
    "clinical parameters using a trained machine learning model. "
    "Please enter the patient details below."
)

st.divider()

# -------------------------------
# User Inputs
# -------------------------------
age = st.number_input("Age", min_value=1, max_value=120, value=45)

sex = st.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

cp = st.number_input("Chest Pain Type (0–3)", min_value=0, max_value=3)
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

restecg = st.number_input("Resting ECG Results (0–2)", min_value=0, max_value=2)
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220)

exang = st.selectbox(
    "Exercise Induced Angina",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, step=0.1)
slope = st.number_input("Slope of Peak Exercise ST Segment (0–2)", min_value=0, max_value=2)
ca = st.number_input("Number of Major Vessels (0–3)", min_value=0, max_value=3)
thal = st.number_input("Thalassemia (1–3)", min_value=1, max_value=3)

st.divider()

# -------------------------------
# Prediction with Confidence
# -------------------------------
if st.button("Predict Heart Disease Risk"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                             restecg, thalach, exang, oldpeak,
                             slope, ca, thal]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(
            f"⚠️ High Risk of Heart Disease Detected\n\n"
            f"Confidence: {probability * 100:.2f}%"
        )
    else:
        st.success(
            f"✅ Low Risk: No Heart Disease Detected\n\n"
            f"Confidence: {(1 - probability) * 100:.2f}%"
        )

# -------------------------------
# Footer
# -------------------------------
st.caption("Machine Learning Project | Heart Disease Prediction System")
