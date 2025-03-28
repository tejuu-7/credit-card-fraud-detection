import pandas as pd
import joblib

# Load model
rf_model = joblib.load("models/fraud_detection_model.pkl")

# Sample new transaction for prediction
new_transaction = pd.DataFrame({
    "Time": [0.5], "V1": [-1.0], "V2": [2.3], "V3": [-0.5], 
    "V4": [0.8], "V5": [-2.0], "V6": [1.1], "V7": [-0.9], 
    "V8": [2.7], "V9": [0.4], "V10": [-1.4], "V11": [0.5], 
    "V12": [-0.7], "V13": [2.1], "V14": [-1.3], "V15": [0.9], 
    "V16": [-0.5], "V17": [1.7], "V18": [-2.2], "V19": [0.6], 
    "V20": [-1.8], "V21": [0.7], "V22": [-0.6], "V23": [1.4], 
    "V24": [-1.9], "V25": [2.5], "V26": [-0.3], "V27": [1.1], 
    "V28": [-2.0], "Amount": [150.0]
})

# Predict fraud
prediction = rf_model.predict(new_transaction)
print("🔍 Fraud Detected!" if prediction[0] == 1 else "✅ Transaction Safe.")
