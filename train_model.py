import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load dataset
df = pd.read_csv("data/sample_credit_data.csv")

# Encode labels
le = LabelEncoder()
df["CreditScore"] = le.fit_transform(df["CreditScore"])   # Good=1, Bad=0

# Features & labels
X = df.drop("CreditScore", axis=1)
y = df["CreditScore"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save Model
os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/credit_model.pkl", "wb"))

print("Model trained & saved in model/credit_model.pkl")
