import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# ---------------------------
# Paths
# ---------------------------
features_path = "C:/Hackathon-AgriAI/image_features.csv"   # Make sure this CSV exists
model_dir = "C:/Hackathon-AgriAI/models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "yield_model.pkl")

# ---------------------------
# Load dataset
# ---------------------------
if not os.path.exists(features_path):
    raise FileNotFoundError(f"{features_path} not found. Run feature extraction first.")

df = pd.read_csv(features_path)

# Check if "actual_yield" exists
if "actual_yield" not in df.columns:
    raise ValueError("❌ Your CSV must contain an 'actual_yield' column as target.")

# ---------------------------
# Define features & target
# ---------------------------
feature_cols = [
    "leaf_count", "mean_area", "total_area",
    "mean_length", "mean_width"
]

missing_cols = [c for c in feature_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"❌ Missing columns in CSV: {missing_cols}")

X = df[feature_cols]
y = df["actual_yield"]

# ---------------------------
# Train model
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------
# Evaluate model
# ---------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained successfully")
print(f"   Test MSE: {mse:.2f}")
print(f"   R² Score: {r2:.2f}")

# ---------------------------
# Save model
# ---------------------------
joblib.dump(model, model_path)
print(f"✅ Model saved to {model_path}")
