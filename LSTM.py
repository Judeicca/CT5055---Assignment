import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from meteostat import Stations, Daily

# Step 1: Data Preprocessing
def preprocess_data(data):
    # Drop any rows with missing values
    data.dropna(inplace=True)
    
    # Normalize the data using Min-Max scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    return data_scaled, scaler

# Step 2: Create sequences for LSTM
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Step 3: LSTM Model Architecture
def create_lstm_model(sequence_length, n_features):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 4: Training Process
def train_model(model, X_train, y_train, epochs, batch_size):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# Step 5: Prediction
def predict_future_values(model, data, scaler, sequence_length, n_features, future_steps):
    future_predictions = []

    # Take the last sequence from the data
    current_sequence = data[-sequence_length:].reshape(1, sequence_length, n_features)

    # Predict future values step by step
    for i in range(future_steps):
        prediction = model.predict(current_sequence)[0]
        future_predictions.append(prediction)

        # Update the current sequence to include the predicted value
        current_sequence = np.append(current_sequence[:, 1:, :], [[prediction]], axis=1)

    # Inverse transform the predictions to get the actual values
    future_predictions = scaler.inverse_transform(future_predictions)
    
    return future_predictions

# Main function
def main():
    # Fetch weather data using Meteostat
    stations = Stations()
    station = stations.find('USW00014732')  # Replace with appropriate station ID
    data = Daily(station, start='2020-01-01', end='2020-12-31').fetch()
    data = data[['tavg']].rename(columns={'tavg': 'Temperature'})

    # Preprocess the data
    data_scaled, scaler = preprocess_data(data)

    # Define parameters
    sequence_length = 10
    epochs = 100
    batch_size = 32
    future_steps = 10

    # Create sequences for LSTM
    X, y = create_sequences(data_scaled, sequence_length)
    n_features = X.shape[2]

    # Split data into train and test sets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Create and train LSTM model
    model = create_lstm_model(sequence_length, n_features)
    train_model(model, X_train, y_train, epochs, batch_size)

    # Predict future values
    future_predictions = predict_future_values(model, data_scaled, scaler, sequence_length, n_features, future_steps)

    # Plot the results
    plt.plot(data.index[-100:], data_scaled[-100:], label='Actual Data')
    plt.plot(np.arange(len(data), len(data) + future_steps), future_predictions, label='Future Predictions')
    plt.xlabel('Time')
    plt.ylabel('Normalized Value')
    plt.title('Future Weather Prediction')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
