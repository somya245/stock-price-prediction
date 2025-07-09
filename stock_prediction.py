import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import math

# Ensure directories exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

def download_data():
    """Download stock data from Yahoo Finance"""
    print("Downloading stock data...")
    data = yf.download("AAPL", start="2010-01-01", end="2023-12-31")
    data = data[['Close']]
    data.to_csv('data/raw/AAPL.csv')
    return data

def preprocess_data(data):
    """Normalize and prepare data for LSTM"""
    print("Preprocessing data...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Save scaler for later use
    import joblib
    joblib.dump(scaler, 'data/processed/scaler.pkl')
    
    # Split data
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - 60:]  # Keep last 60 days for sequence
    
    # Create sequences
    def create_sequences(data, time_steps=60):
        X, y = [], []
        for i in range(time_steps, len(data)):
            X.append(data[i-time_steps:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    X_train, y_train = create_sequences(train_data)
    X_test, y_test = create_sequences(test_data)
    
    # Reshape for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    return X_train, y_train, X_test, y_test, scaler

def build_model(input_shape):
    """Build LSTM model"""
    print("Building model...")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train):
    """Train the LSTM model"""
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=100,
        validation_split=0.1
    )
    model.save('models/lstm_model.h5')
    return history

def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate model performance"""
    print("Evaluating model...")
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    
    # Calculate RMSE
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = math.sqrt(mean_squared_error(y_test_actual, predictions))
    print(f"RMSE: {rmse}")
    
    return predictions

def plot_results(data, train_size, predictions):
    """Plot actual vs predicted prices"""
    print("Generating plot...")
    train = data[:train_size]
    valid = data[train_size:]
    valid['Predictions'] = predictions
    
    plt.figure(figsize=(16,8))
    plt.title('Stock Price Prediction using LSTM')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Actual', 'Predictions'], loc='lower right')
    plt.savefig('stock_prediction_plot.png')
    plt.show()

def predict_future_price(model, scaler, data):
    """Predict next day's price"""
    last_60_days = data[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)
    X_future = np.reshape(last_60_days_scaled, (1, 60, 1))
    future_price = model.predict(X_future)
    future_price = scaler.inverse_transform(future_price)
    print(f"Predicted next day closing price: ${future_price[0][0]:.2f}")

def main():
    # Step 1: Download data
    data = download_data()
    
    # Step 2: Preprocess data
    X_train, y_train, X_test, y_test, scaler = preprocess_data(data)
    
    # Step 3: Build and train model
    model = build_model(input_shape=(X_train.shape[1], 1))
    history = train_model(model, X_train, y_train)
    
    # Step 4: Evaluate model
    predictions = evaluate_model(model, X_test, y_test, scaler)
    
    # Step 5: Plot results
    train_size = int(len(data) * 0.8)
    plot_results(data, train_size, predictions)
    
    # Step 6: Future prediction
    predict_future_price(model, scaler, data)

if __name__ == "__main__":
    main()