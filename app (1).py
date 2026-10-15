import pickle
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

MODEL_PATH = Path(__file__).parent / "heart_svc_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("❤️ Heart Disease Prediction System")
st.markdown("Enter patient details below to predict the risk of heart disease.")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
with col2:
    cp = st.selectbox("Chest Pain Type (CP)", options=[1, 2, 3, 4],
                      format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal Pain", 4: "Asymptomatic"}[x])
    thalach = st.number_input("Max Heart Rate (thalach)", min_value=50, max_value=230, value=150)

if st.button("Predict Risk", type="primary", use_container_width=True):
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'thalach': [thalach]
    })

    prediction = model.predict(input_data)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ HIGH RISK: Heart Disease Detected")
    else:
        st.success("### ✅ LOW RISK: No Heart Disease Detected")
