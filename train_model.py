import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # For saving the trained model

# Load sensor data
df = pd.read_csv("sensor_data.csv")

# Define features and labels
X = df[["temperature", "vibration", "pressure"]]  # Features
y = df["failure"]  # Target variable

# Split data: Train on first 23 days, Test on last 7 days
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# Train a RandomForest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with Accuracy: {accuracy:.2f}")

# Save model for later use
joblib.dump(model, "predictive_model.pkl")
print("✅ Model saved as predictive_model.pkl")
