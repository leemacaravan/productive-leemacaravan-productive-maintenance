import numpy as np
import pandas as pd

# Generate timestamps (30 days of hourly data)
time_range = pd.date_range(start="2024-01-01", periods=720, freq="H")

# Generate synthetic sensor data
np.random.seed(42)
temperature = np.random.normal(40, 5, len(time_range))  # Avg 40°C
vibration = np.random.normal(5, 2, len(time_range))  # Avg 5 m/s² vibration
pressure = np.random.normal(100, 10, len(time_range))  # Avg 100 kPa pressure

# Simulate failures (higher values = risk of failure)
failure = np.zeros(len(time_range))
failure[-168:] = np.random.choice([0, 1], size=168, p=[0.85, 0.15])  # 15% failure rate in last 7 days

# Create DataFrame
df = pd.DataFrame({"time": time_range, "temperature": temperature, "vibration": vibration, "pressure": pressure, "failure": failure})

# Save to CSV
df.to_csv("sensor_data.csv", index=False)
print("✅ Sensor data generated and saved as sensor_data.csv")
