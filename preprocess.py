import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("dataset/creditcard.csv")

# Feature scaling
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])
df["Time"] = scaler.fit_transform(df[["Time"]])

# Split dataset
X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save processed data
X_train.to_csv("dataset/X_train.csv", index=False)
X_test.to_csv("dataset/X_test.csv", index=False)
y_train.to_csv("dataset/y_train.csv", index=False)
y_test.to_csv("dataset/y_test.csv", index=False)

print("✅ Data Preprocessing Completed & Saved.")
