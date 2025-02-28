import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_lstm_model(input_shape):
    """Define and compile an LSTM model for crypto price prediction."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)  # Output layer (predicting closing price)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_prepare_data(file_path="preprocessed_data.csv", include_real_time=True):
    """Load historical data and preprocess real-time BTC/USDT & ETH/USDT data if needed."""
    df = pd.read_csv(file_path)

    if include_real_time:
        real_time_df = preprocess_real_time_data()

        if not real_time_df.empty:
            df = pd.concat([df, real_time_df], ignore_index=True)

    # ✅ Ensure all timestamps are in datetime format before sorting
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # ✅ Drop any rows where timestamp conversion failed
    df = df.dropna(subset=["timestamp"])

    # ✅ Sort data by timestamp after ensuring uniform datetime format
    df = df.sort_values(by="timestamp")

    # Use only the 'close' price for training (BTC & ETH separately)
    btc_data = df[df["symbol"] == "BTC/USDT"]["close"].values.reshape(-1, 1)
    eth_data = df[df["symbol"] == "ETH/USDT"]["close"].values.reshape(-1, 1)

    # Normalize data
    scaler_btc = MinMaxScaler(feature_range=(0, 1))
    scaler_eth = MinMaxScaler(feature_range=(0, 1))

    scaled_btc_data = scaler_btc.fit_transform(btc_data)
    scaled_eth_data = scaler_eth.fit_transform(eth_data)

    # Create sequences for LSTM
    def create_sequences(scaled_data):
        sequence_length = 60
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(X), np.array(y)

    X_btc, y_btc = create_sequences(scaled_btc_data)
    X_eth, y_eth = create_sequences(scaled_eth_data)

    # Reshape for LSTM
    X_btc = np.reshape(X_btc, (X_btc.shape[0], X_btc.shape[1], 1))
    X_eth = np.reshape(X_eth, (X_eth.shape[0], X_eth.shape[1], 1))

    # Split into training and testing sets
    X_train_btc, X_test_btc, y_train_btc, y_test_btc = train_test_split(X_btc, y_btc, test_size=0.2, random_state=42, shuffle=False)
    X_train_eth, X_test_eth, y_train_eth, y_test_eth = train_test_split(X_eth, y_eth, test_size=0.2, random_state=42, shuffle=False)

    print(f"\n✅ BTC Data Loaded: {X_train_btc.shape[0]} training samples, {X_test_btc.shape[0]} test samples")
    print(f"✅ ETH Data Loaded: {X_train_eth.shape[0]} training samples, {X_test_eth.shape[0]} test samples")

    return X_train_btc, X_test_btc, y_train_btc, y_test_btc, scaler_btc, X_train_eth, X_test_eth, y_train_eth, y_test_eth, scaler_eth


def train_lstm_model(model, X_train, y_train, epochs=20, batch_size=32):
    """Train the LSTM model using the training dataset."""
    print("\n🚀 Training LSTM Model...")
    
    history = model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=0.2, 
        verbose=1
    )
    
    print("\n✅ Training Complete!")
    return history

from pymongo import MongoClient

def preprocess_real_time_data():
    """Fetch, clean, and prepare real-time BTC/USDT & ETH/USDT data from MongoDB."""
    client = MongoClient("mongodb://localhost:27017/")
    db = client["crypto_data"]
    collection = db["Exchange Prices"]

    real_time_symbols = ["BTC/USDT", "ETH/USDT"]
    real_time_data = []

    for symbol in real_time_symbols:
        latest_data = collection.find({"token_pair": symbol}, sort=[("timestamp", -1)])
        real_time_df = pd.DataFrame(list(latest_data))

        if not real_time_df.empty:
            real_time_df = real_time_df[["timestamp", "prices"]]
            real_time_df["close"] = real_time_df["prices"].apply(lambda x: x.get("Binance", None))

            # Drop rows where price data is missing
            real_time_df = real_time_df.drop(columns=["prices"]).dropna()

            real_time_df["symbol"] = symbol
            real_time_df["source"] = "Real-Time"

            # Convert timestamp to datetime for proper sorting
            real_time_df["timestamp"] = pd.to_datetime(real_time_df["timestamp"])

            real_time_data.append(real_time_df)

    if real_time_data:
        real_time_df_combined = pd.concat(real_time_data, ignore_index=True).sort_values(by="timestamp")
        print(f"\n✅ Preprocessed Real-Time Data: {real_time_df_combined.shape[0]} records processed.")
        return real_time_df_combined
    else:
        print("\n⚠️ No valid real-time data found.")
        return pd.DataFrame()


def get_latest_real_time_data(symbol="BTC/USDT"):
    """Fetch the latest real-time price from MongoDB."""
    client = MongoClient("mongodb://localhost:27017/")
    db = client["crypto_data"]
    collection = db["Exchange Prices"]

    latest_data = collection.find_one({"token_pair": symbol}, sort=[("timestamp", -1)])
    
    if latest_data:
        latest_prices = latest_data.get("prices", {})
        if "Binance" in latest_prices:
            real_time_price = latest_prices["Binance"]
            print(f"📊 Latest Real-Time Price from MongoDB for {symbol}: {real_time_price}")
            return real_time_price
    print(f"⚠️ No real-time data found for {symbol} in MongoDB")
    return None

def predict_real_time_price_from_mongo(model, scaler, symbol="BTC/USDT"):
    """Fetch real-time data, append it to the last 59 historical prices, and predict the next price."""
    real_time_price = get_latest_real_time_data(symbol)

    if real_time_price is None:
        print(f"⚠️ No real-time data available for {symbol}")
        return None

    # Load the last 59 historical prices for this symbol
    df = pd.read_csv("preprocessed_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])  # Ensure no NaN timestamps
    df = df.sort_values(by="timestamp")

    # Filter historical data for the given symbol
    last_59_prices = df[df["symbol"] == symbol]["close"].values[-59:].reshape(-1, 1)

    if len(last_59_prices) < 59:
        print(f"⚠️ Not enough historical data for {symbol} to make a prediction.")
        return None

    # Normalize historical prices
    scaled_last_59 = scaler.transform(last_59_prices)

    # Normalize the real-time price
    scaled_real_time = scaler.transform([[real_time_price]])[0][0]

    # Create final sequence (last 59 historical prices + 1 real-time price)
    X_real_time = np.append(scaled_last_59, scaled_real_time).reshape(1, 60, 1)

    # Make prediction
    predicted_scaled_price = model.predict(X_real_time)[0][0]

    # Convert back to actual price
    predicted_price = scaler.inverse_transform([[predicted_scaled_price]])[0][0]

    print(f"🚀 Predicted Future Price for {symbol}: {predicted_price}")
    return predicted_price


if __name__ == "__main__":
    input_shape = (60, 1)
    
    # Load and merge real-time BTC & ETH data with historical data
    X_train_btc, X_test_btc, y_train_btc, y_test_btc, scaler_btc, X_train_eth, X_test_eth, y_train_eth, y_test_eth, scaler_eth = load_and_prepare_data(include_real_time=True)

    # Train BTC Model
    print("\n🚀 Training BTC/USDT Model...")
    model_btc = create_lstm_model(input_shape)
    history_btc = train_lstm_model(model_btc, X_train_btc, y_train_btc)

    # Train ETH Model
    print("\n🚀 Training ETH/USDT Model...")
    model_eth = create_lstm_model(input_shape)
    history_eth = train_lstm_model(model_eth, X_train_eth, y_train_eth)

    # Test real-time prediction for BTC
    print("\n🔮 Predicting BTC/USDT Future Price...")
    predict_real_time_price_from_mongo(model_btc, scaler_btc, "BTC/USDT")

    # Test real-time prediction for ETH
    print("\n🔮 Predicting ETH/USDT Future Price...")
    predict_real_time_price_from_mongo(model_eth, scaler_eth, "ETH/USDT")