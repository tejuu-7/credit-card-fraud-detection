# Credit Card Fraud Detection - GrowthLink Internship

## 📌 Project Overview

This project is part of the **GrowthLink Internship** and focuses on detecting fraudulent credit card transactions using machine learning. The system is trained on a dataset containing genuine and fraudulent transactions and aims to improve fraud detection accuracy while minimizing false positives.

## 📂 Project Structure

```
GrowthLink-Fraud-Detection/
│── dataset/                # Contains the dataset files
│   ├── creditcard.csv      # Original dataset (not included due to size limit)
│   ├── X_train.csv         # Processed training features
│   ├── y_train.csv         # Processed training labels
│
│── models/                 # Trained models
│   ├── fraud_detection_model.pkl   # Saved Random Forest model
│
│── scripts/                # Python scripts for different stages of execution
│   ├── preprocess.py       # Data preprocessing and feature engineering
│   ├── train.py            # Model training with SMOTE balancing
│   ├── evaluate.py         # Model evaluation with accuracy, precision, recall
│   ├── predict.py          # Fraud prediction for new transactions
│
│── fraud_detection.py       # Main execution file
│── confusion_matrix.png     # Visual representation of model performance
│── requirements.txt         # Dependencies required for the project
│── README.md                # Project documentation
```

## 📊 Dataset Details

- **Source:** Kaggle (Credit Card Fraud Detection dataset)
- **Features:** Time, Amount, 28 anonymized PCA components
- **Target Variable:** `Class` (0 = Non-Fraud, 1 = Fraud)
- **Imbalance:** 99.83% non-fraud, 0.17% fraud transactions

## ⚙️ Installation & Setup

### **1️⃣ Create a Virtual Environment**

```bash
python -m venv fraud_env
source fraud_env/bin/activate   # On macOS/Linux
fraud_env\Scripts\activate     # On Windows
```

### **2️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

## 🚀 Execution Steps

### **Step 1: Data Preprocessing**

```bash
python scripts/preprocess.py
```

✅ This step cleans and prepares the dataset.

### **Step 2: Model Training**

```bash
python scripts/train.py
```

✅ This step trains a **Random Forest model** with SMOTE balancing.

### **Step 3: Model Evaluation**

```bash
python scripts/evaluate.py
```

✅ Expected Output:

```
Accuracy: 0.9996
Precision: 0.9302
Recall: 0.8163
F1 Score: 0.8696
```

### **Step 4: Fraud Prediction**

```bash
python scripts/predict.py dataset/sample_transaction.csv
```

✅ Expected Output:

```
✅ Transaction Safe.
```

## 🛠 Future Enhancements

- Implement **Deep Learning (LSTM)** for improved accuracy
- Integrate **real-time fraud detection API**
- Use **Explainable AI (SHAP, LIME)** to interpret predictions

## 🤝 Contribution

This project was completed as part of the **GrowthLink Internship**. Contributions and suggestions are welcome!

## 📝 License

This project is open-source and available under the **MIT License**.


