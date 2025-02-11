import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load sensor data
df = pd.read_csv("sensor_data.csv")

# Load trained ML model
model = joblib.load("predictive_model.pkl")

# Predict failures
df["predicted_failure"] = model.predict(df[["temperature", "vibration", "pressure"]])

# Streamlit UI
st.title("🚀 Predictive Maintenance Dashboard")
st.write("Monitor machine health and predict failures.")

# Display sensor data
st.subheader("📊 Sensor Data Overview")
st.line_chart(df[["temperature", "vibration", "pressure"]])

# Show predicted failures
failure_count = df["predicted_failure"].sum()
st.subheader(f"⚠ Predicted Failures in the Next 7 Days: {failure_count}")

# Show sample predictions
st.write("📋 Sample Predictions:")
st.dataframe(df.tail(50))

# Visualize sensor readings
st.subheader("📈 Sensor Data Trends")
fig, ax = plt.subplots(3, 1, figsize=(10, 6))
ax[0].plot(df["temperature"], color="red", label="Temperature")
ax[1].plot(df["vibration"], color="blue", label="Vibration")
ax[2].plot(df["pressure"], color="green", label="Pressure")
ax[0].set_title("Temperature Over Time")
ax[1].set_title("Vibration Over Time")
ax[2].set_title("Pressure Over Time")
st.pyplot(fig)
