import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --- Create model folder ---
os.makedirs("model", exist_ok=True)

# --- Load dataset ---
df = pd.read_csv("data/Rainfall.csv")

# --- Clean columns ---
df.columns = df.columns.str.strip()

# --- Convert yes/no to 1/0 ---
df["rainfall"] = df["rainfall"].map({"yes": 1, "no": 0})

# --- Convert all other columns to numeric ---
for col in df.columns:
    if col != "rainfall":
        df[col] = pd.to_numeric(df[col], errors="ignore")

# --- Fill missing values ---
df = df.fillna(df.mean(numeric_only=True))

# --- Split into X and y ---
X = df.drop("rainfall", axis=1)
y = df["rainfall"]

# --- Scale Inputs ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train-test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# --- Train Model ---
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# --- Save model + scaler ---
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("🎉 SUCCESS! Model and scaler created:")
print("✔ model/model.pkl")
print("✔ model/scaler.pkl")
