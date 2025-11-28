import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# 1. Setup paths
ROOT = Path(__file__).parent.resolve()
MODEL_PATH = ROOT / "outputs" / "model_logistic.joblib"
SCALER_PATH = ROOT / "outputs" / "scaler.joblib"

def predict_demo():
    print(f"Loading model from {MODEL_PATH}...")
    
    # 2. Load the binary files
    # This is why you can't open them in a text editor - they are serialized Python objects.
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        print("Error: Model files not found. Did you run main.py?")
        return

    print("Model and Scaler loaded successfully!")

    # 3. Create a dummy patient (already processed/encoded)
    # In a real app, you would take raw input (Age=60, Sex=Male) and run it through 
    # the same cleaning/encoding steps as in preprocessing.py.
    # Here, we just create a random feature vector of the correct size (19 features) to demonstrate.
    
    # The model expects 18 features.
    num_features = 18 
    dummy_patient_data = np.random.rand(1, num_features)
    
    # 4. Scale the data (using the loaded scaler)
    # Note: The scaler expects the same number of features used during training.
    # If the feature count doesn't match, this might fail or give a warning, 
    # because our dummy data isn't perfectly aligned with the training columns.
    # For this demo, we'll skip scaling if dimensions don't match, just to show the prediction part.
    
    try:
        patient_scaled = scaler.transform(dummy_patient_data)
    except ValueError:
        print("Note: Skipping scaler for this dummy demo due to feature mismatch.")
        patient_scaled = dummy_patient_data

    # 5. Make a prediction
    prediction = model.predict(patient_scaled)
    probability = model.predict_proba(patient_scaled)

    print("\n--- Prediction Result ---")
    print(f"Predicted Class: {prediction[0]} (0=No Disease, 1=Disease)")
    print(f"Confidence: {probability[0][prediction[0]]:.2f}")
    print("-------------------------")
    print("\nSuccess! This demonstrates how to use the .joblib files in Python.")

if __name__ == "__main__":
    predict_demo()
