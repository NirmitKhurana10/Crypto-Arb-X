from pymongo import MongoClient
import pandas as pd
import numpy as np

# MongoDB connection setup
def get_mongo_connection(uri="mongodb://localhost:27017/", db_name="crypto_data"):
    """Connect to MongoDB and return the database object."""
    client = MongoClient(uri)
    db = client[db_name]
    return db

def fetch_historical_data():
    """Retrieve historical crypto data from all collections and combine into a single DataFrame."""
    db = get_mongo_connection()
    collections = ["BTC_USDT", "ETH_USDT"]
    
    all_data = []
    for collection_name in collections:
        collection = db[collection_name]
        cursor = collection.find({}, {"_id": 0})  # Exclude MongoDB _id field
        df = pd.DataFrame(list(cursor))
        
        if not df.empty:
            df["source"] = collection_name  # Add a column to identify source collection
            all_data.append(df)
            print(f"✅ Retrieved {len(df)} records from {collection_name}", flush=True)
            print(f"\n📊 Sample Data from {collection_name}:\n{df.head(5)}\n", flush=True)  # Print first 5 rows
    
    # Combine all data into one DataFrame
    final_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    return final_df

def clean_data(df):
    """Handle missing values and remove duplicates."""
    df = df.drop_duplicates()
    df = df.dropna()
    print("✅ Data Cleaning Complete!", flush=True)
    return df

def normalize_data(df, columns):
    """Normalize specified columns using Min-Max scaling."""
    for col in columns:
        if col in df.columns:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    print("✅ Data Normalization Complete!", flush=True)
    return df

def save_preprocessed_data(df, filename="preprocessed_data.csv"):
    """Save the preprocessed data to a CSV file."""
    df.to_csv(filename, index=False)
    print(f"\n✅ Preprocessed data saved as {filename}", flush=True)

def preprocess_data():
    """Fetch, clean, normalize, and save crypto data from MongoDB."""
    df = fetch_historical_data()
    
    if df.empty:
        print("⚠️ No data found in MongoDB collections!", flush=True)
        return None

    print("\n📊 Raw Data Sample Before Cleaning:\n", df.head(5), flush=True)

    # Data Cleaning
    df = clean_data(df)

    # Define columns to normalize
    columns_to_normalize = ["open", "high", "low", "close", "volume"]
    df = normalize_data(df, columns_to_normalize)

    # **Check record counts by symbol**
    print("\n📊 Data Breakdown by Symbol Before Saving:\n", df["symbol"].value_counts(), flush=True)

    print("\n📊 Normalized Sample Data:\n", df.head(10), flush=True)  # Print first 10 rows of normalized data
    print("\n📊 Dataset Summary:\n", flush=True)
    print(df.info(), flush=True)  # Print dataset summary

    # Save the processed data
    save_preprocessed_data(df)

    return df

# Example test run
if __name__ == "__main__":
    df = preprocess_data()
