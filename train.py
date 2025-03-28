import pandas as pd
import joblib
import os
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Load preprocessed data
X_train = pd.read_csv("dataset/X_train.csv")
y_train = pd.read_csv("dataset/y_train.csv")

# Fix DataConversionWarning: Convert y_train to a 1D array
y_train = y_train.values.ravel()  

# Handle class imbalance using SMOTE
smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train the model with parallel execution for faster training
rf_model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train_resampled, y_train_resampled)

# Ensure 'models/' directory exists before saving
os.makedirs("models", exist_ok=True)

# Save the trained model
joblib.dump(rf_model, "models/fraud_detection_model.pkl")
print("✅ Model Trained & Saved Successfully.")
