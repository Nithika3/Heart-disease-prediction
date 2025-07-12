import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

# ---------- 1. Background Image ----------
def set_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_bg_from_local("img.png")

# ---------- 2. Text Styling ----------
st.markdown("""
<style>
html, body, [class*="css"] {
    color: black !important;
}
.stNumberInput label,
.stSelectbox label,
.stTextInput label,
.stSlider label {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- 3. Navigation Setup ----------
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to_predict():
    st.session_state.page = "predict"

# ---------- 4. Centered Widget Helper ----------
def center_input(widget_func, *args, **kwargs):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        return widget_func(*args, **kwargs)

# ---------- 5. Home Page ----------
if st.session_state.page == "home":
    st.markdown("<h1 style='text-align: center;color:black;'>❤️ Welcome to the Heart Disease Prediction App</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;color:black;'>A Machine Learning Project to Assess Cardiac Risk</h3>", unsafe_allow_html=True)

    st.markdown("""
        <div style='text-align: center; font-size:18px;color:black'>
            This tool helps estimate your risk of heart disease using key health indicators.
            <br><br>
            Click the button below to start the prediction.
        </div>
    """, unsafe_allow_html=True)

    if center_input(st.button, "🔍 Start Prediction"):
        go_to_predict()

# ---------- 6. Prediction Page ----------
elif st.session_state.page == "predict":
    st.markdown("<h1 style='text-align: center;color:black;'>❤️ Heart Disease Prediction</h1>", unsafe_allow_html=True)

    # --- Model Selection ---
    model_choice = center_input(st.selectbox, "Choose Prediction Algorithm", ["XGBoost", "Random Forest", "Logistic Regression"])

    # Load corresponding model
    model_files = {
        "XGBoost": "xgboost_model.joblib",
        "Random Forest": "random_forest_model.joblib",
        "Logistic Regression": "logistic_model.joblib"
    }

    model = joblib.load(model_files[model_choice])
    preprocessor = joblib.load("preprocessor.joblib")

    # --- Input Fields ---
    age = center_input(st.number_input, "Age", min_value=1, max_value=120, value=50)
    sex = center_input(st.selectbox, "Sex (0: Female, 1: Male)", [0, 1])
    chest_pain = center_input(st.selectbox, "Chest Pain Type", [0, 1, 2, 3])
    bp = center_input(st.number_input, "Blood Pressure (BP)", min_value=50, max_value=250, value=120)
    cholesterol = center_input(st.number_input, "Cholesterol", min_value=100, max_value=600, value=200)
    fbs = center_input(st.selectbox, "FBS Over 120 (0: No, 1: Yes)", [0, 1])
    ekg = center_input(st.selectbox, "EKG Results", [0, 1, 2])
    max_hr = center_input(st.number_input, "Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
    exercise_angina = center_input(st.selectbox, "Exercise Angina (0: No, 1: Yes)", [0, 1])
    st_depression = center_input(st.number_input, "ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope_st = center_input(st.selectbox, "Slope of ST", [0, 1, 2])
    num_vessels = center_input(st.selectbox, "Number of Vessels Fluro", [0, 1, 2, 3])
    thallium = center_input(st.selectbox, "Thallium", [0, 1, 2, 3])

    input_df = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Chest pain type': [chest_pain],
        'BP': [bp],
        'Cholesterol': [cholesterol],
        'FBS over 120': [fbs],
        'EKG results': [ekg],
        'Max HR': [max_hr],
        'Exercise angina': [exercise_angina],
        'ST depression': [st_depression],
        'Slope of ST': [slope_st],
        'Number of vessels fluro': [num_vessels],
        'Thallium': [thallium]
    })

    processed_input = preprocessor.transform(input_df)

    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        if st.button("🧠 Predict"):
            prediction = model.predict(processed_input)
            probability = model.predict_proba(processed_input)[0][1]

            st.markdown("---")
            if prediction[0] == 1:
                st.markdown(f"""
                <div style='background-color:#ffcccc; padding:20px; border-radius:10px'>
                    <h4 style='color:black;'>🚨 <b>High Risk Detected!</b></h4>
                    <p style='color:black;'>Risk Score: <b>{probability*100:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color:#ccffcc; padding:20px; border-radius:10px'>
                    <h4 style='color:black;'>✅ <b>Low Risk Detected</b></h4>
                    <p style='color:black;'>Risk Score: <b>{probability*100:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
