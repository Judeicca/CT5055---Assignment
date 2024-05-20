# Importing the required libraries
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from meteostat import Daily
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

# Fetching daily temperature data from the Meteostat API
station_id = '03726'  # Station ID for Bristol Weather Centre
start = datetime(2019, 1, 1)
end = datetime(2023, 12, 31)
data = Daily(station_id, start, end)
data = data.fetch()

# Preprocessing the data
data.index = pd.to_datetime(data.index)
data['day_of_year'] = data.index.dayofyear

# Calculating the average daily temperatures, also dropping missing values
average_daily_temps = data.groupby('day_of_year')['tavg'].mean().dropna()

# Interpolating missing values
average_daily_temps = average_daily_temps.interpolate(method='linear')

# Normalize the average temperatures
scaler = MinMaxScaler()
average_temps_scaled = scaler.fit_transform(average_daily_temps.values.reshape(-1, 1))

# A function to create sequences of data suitable for training an LSTM model
def create_sequences(data, sequence_length):
    xs = [] # List to store input sequences
    ys = [] # List to store output values

    # Iterate over the data to create sequences
    for i in range(len(data) - sequence_length):
        xs.append(data[i:(i + sequence_length)]) 
        ys.append(data[i + sequence_length]) 
    return np.array(xs), np.array(ys) # Return the input and output sequences

sequence_length = 4  # Using the last 4 years to predict the next year

# Create sequences for training the LSTM model
X, y = create_sequences(average_temps_scaled, sequence_length)

# Splitting data into training and testing sets
split = int(0.8 * len(X)) # 80% training, 20% testing
X_train, X_test = X[:split], X[split:] # Split the input sequences
y_train, y_test = y[:split], y[split:] # Split the output values

# Reshape the input sequences for LSTM
model = Sequential([
    LSTM(50, input_shape=(sequence_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error') # Compile the model

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=1000, validation_split=0.1)

# Evaluate the model on the testing set
y_pred = model.predict(X_test) # Predict the output values
mse = mean_squared_error(y_test, y_pred) # Calculate the Mean Squared Error
print("Mean Squared Error:", mse)

# A function to predict the temperature for a given day
def predict_for_day(model, sequence):
    sequence_reshaped = sequence.reshape(1, sequence_length, 1) # Reshape the input sequence
    predicted_temperature = model.predict(sequence_reshaped) # Predict the temperature
    return scaler.inverse_transform(predicted_temperature)[0, 0]  # Flatten the output

last_sequence = average_temps_scaled[-sequence_length:] # Last sequence of the data

# Predicting temperatures for 2024 by iterating over the data and predicting each day 
predicted_temperatures_2024 = np.array([predict_for_day(model, average_temps_scaled[i:i+sequence_length]) for i in range(len(average_temps_scaled) - sequence_length)]) 

# Dates for plotting
dates_2024 = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(len(predicted_temperatures_2024))]

# Plot the predictions alongside historical data
plt.figure(figsize=(15, 7))
plt.plot(data.index, data['tavg'], label='Historical Avg Temps (2019-2023)', color='blue')
plt.plot(dates_2024, predicted_temperatures_2024, label='Predicted Avg Temps for 2024', color='red', linestyle='--')
plt.title('Historical and Predicted Daily Average Temperatures for Bristol (2019-2024)')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()
