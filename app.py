import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Setup paths
ROOT = Path(__file__).parent.resolve()
MODEL_PATH = ROOT / "outputs" / "model_logistic.joblib"
SCALER_PATH = ROOT / "outputs" / "scaler.joblib"

# Page Config
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# Load Assets
@st.cache_resource
def load_assets():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_assets()

# Header
st.title("❤️ Heart Disease Risk Predictor")
st.markdown("Enter your health details below to get a risk assessment.")

if model is None:
    st.error("Error: Model files not found. Please run `main.py` first to train the model.")
    st.stop()

# Input Form
with st.form("prediction_form"):
    st.subheader("Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
        
    with col2:
        restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=0.0)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
        ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"])

    submit_btn = st.form_submit_button("Predict Risk")

if submit_btn:
    # Preprocessing Input
    # We need to match the exact feature columns used in training.
    # Since we used pd.get_dummies, we need to manually reconstruct the dataframe.
    
    # 1. Create a dictionary with raw inputs
    input_data = {
        'age': age,
        'trestbps': trestbps,
        'chol': chol,
        'thalach': thalach,
        'oldpeak': oldpeak,
        'ca': ca,
        'sex_Male': 1 if sex == "Male" else 0,
        'cp_atypical angina': 1 if cp == "Atypical Angina" else 0,
        'cp_non-anginal': 1 if cp == "Non-anginal Pain" else 0,
        'cp_typical angina': 1 if cp == "Typical Angina" else 0,
        'fbs_True': 1 if fbs == "True" else 0,
        'restecg_lv hypertrophy': 1 if restecg == "Left Ventricular Hypertrophy" else 0,
        'restecg_normal': 1 if restecg == "Normal" else 0,
        'restecg_st-t abnormality': 1 if restecg == "ST-T Wave Abnormality" else 0,
        'exang_True': 1 if exang == "Yes" else 0,
        'slope_flat': 1 if slope == "Flat" else 0,
        'slope_upsloping': 1 if slope == "Upsloping" else 0,
        'thal_fixed defect': 1 if thal == "Fixed Defect" else 0,
        'thal_normal': 1 if thal == "Normal" else 0,
        'thal_reversable defect': 1 if thal == "Reversable Defect" else 0
    }
    
    # Note: This manual mapping is fragile if feature names change. 
    # Ideally, we would save the column names during training.
    # For now, we will try to align with the 18 features expected by the model.
    
    # Let's check the expected features from the scaler if possible, or just build the DF.
    # Based on previous error, model expects 18 features.
    # Let's construct the DataFrame.
    
    # We need to know the EXACT column order. 
    # Best practice: Save feature names in training.
    # Workaround: We will use the feature_importance.csv to get the names if available, 
    # or just try to match the common one-hot encoding output.
    
    # Let's try to load feature names from the csv we generated.
    # Define the exact feature order expected by the model
    feature_names = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 'sex_Male', 'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina', 'fbs_True', 'restecg_normal', 'restecg_st-t abnormality', 'exang_True', 'slope_flat', 'slope_upsloping', 'thal_normal', 'thal_reversable defect']
    
    # Create a zero-filled dataframe with these columns
    df_input = pd.DataFrame(0, index=[0], columns=feature_names)
        
    # Fill known values
    # Numeric
    df_input['age'] = age
    df_input['trestbps'] = trestbps
    df_input['chol'] = chol
    df_input['thalch'] = thalach
    df_input['oldpeak'] = oldpeak
    df_input['ca'] = ca
    
    # Categorical (One-Hot)
    # We need to map our UI selections to the specific column names like 'sex_Male'
    
    # Helper to set value if column exists
    def set_col(name, val):
        if name in df_input.columns:
            df_input[name] = val
            
    set_col('sex_Male', 1 if sex == "Male" else 0)
    
    # CP
    if cp == "Typical Angina": set_col('cp_typical angina', 1)
    elif cp == "Atypical Angina": set_col('cp_atypical angina', 1)
    elif cp == "Non-anginal Pain": set_col('cp_non-anginal', 1)
    # Asymptomatic is usually the dropped column (base case)
    
    set_col('fbs_True', 1 if fbs == "True" else 0)
    
    # RestECG
    if restecg == "Normal": set_col('restecg_normal', 1)
    elif restecg == "Left Ventricular Hypertrophy": set_col('restecg_lv hypertrophy', 1)
    # ST-T is likely the other one
    
    set_col('exang_True', 1 if exang == "Yes" else 0)
    
    # Slope
    if slope == "Flat": set_col('slope_flat', 1)
    elif slope == "Upsloping": set_col('slope_upsloping', 1)
    
    # Thal
    if thal == "Normal": set_col('thal_normal', 1)
    elif thal == "Reversable Defect": set_col('thal_reversable defect', 1)
    elif thal == "Fixed Defect": set_col('thal_fixed defect', 1)

    # Scale
    X_scaled = scaler.transform(df_input)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]
    
    st.divider()
    if prediction == 1:
        st.error(f"**High Risk Detected** (Confidence: {prob:.2%})")
        st.warning("Please consult a cardiologist for further evaluation.")
    else:
        st.success(f"**Low Risk** (Confidence: {1-prob:.2%})")
        st.info("Keep up the healthy lifestyle!")
            
