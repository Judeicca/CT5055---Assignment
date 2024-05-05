import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from meteostat import Daily
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

# Step 1: Data Collection with a Specific Station ID
station_id = '03726'  # Station ID for Bristol Weather Centre
start = datetime(2019, 1, 1)
end = datetime(2023, 12, 31)
data = Daily(station_id, start, end)
data = data.fetch()

# Ensure data is indexed by date and compute the day of the year
data.index = pd.to_datetime(data.index)
data['day_of_year'] = data.index.dayofyear

# Group by 'day_of_year' to calculate average temperatures for each day across years
average_daily_temps = data.groupby('day_of_year')['tavg'].mean().dropna()

# Handle missing data
average_daily_temps = average_daily_temps.interpolate(method='linear')

# Normalize the average temperatures
scaler = MinMaxScaler()
average_temps_scaled = scaler.fit_transform(average_daily_temps.values.reshape(-1, 1))

# Define a function to create sequences
def create_sequences(data, sequence_length):
    xs = []
    ys = []
    for i in range(len(data) - sequence_length):
        xs.append(data[i:(i + sequence_length)])
        ys.append(data[i + sequence_length])
    return np.array(xs), np.array(ys)

sequence_length = 4  # Using the last 4 years to predict the next year

# Creating sequences of data suitable for training an LSTM model
X, y = create_sequences(average_temps_scaled, sequence_length)

# Splitting data into training and testing sets (80% training, 20% testing)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define the LSTM model
model = Sequential([
    LSTM(50, input_shape=(sequence_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=1000, validation_split=0.1)

# Evaluate the model on the testing set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Function to predict the temperatures for an entire year
def predict_for_day(model, sequence):
    sequence_reshaped = sequence.reshape(1, sequence_length, 1)
    predicted_temperature = model.predict(sequence_reshaped)
    return scaler.inverse_transform(predicted_temperature)[0, 0]  # Flatten the output

# Prepare the last sequence from 2023 data
last_sequence = average_temps_scaled[-sequence_length:]

# Predict temperatures for every day of 2024
predicted_temperatures_2024 = np.array([predict_for_day(model, average_temps_scaled[i:i+sequence_length]) for i in range(len(average_temps_scaled) - sequence_length)])

# Dates for plotting
dates_2024 = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(len(predicted_temperatures_2024))]

# Plot the predictions alongside historical data
plt.figure(figsize=(15, 7))
plt.plot(data.index, data['tavg'], label='Historical Avg Temps (2019-2023)', color='blue')
plt.plot(dates_2024, predicted_temperatures_2024, label='Predicted Avg Temps for 2024', color='red', linestyle='--')
plt.title('Historical and Predicted Daily Average Temperatures')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()
