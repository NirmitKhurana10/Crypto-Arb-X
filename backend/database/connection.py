from pymongo import MongoClient

try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["crypto_data"]
    print("✅ Connected to MongoDB successfully!")
except Exception as e:
    print(f"❌ MongoDB Connection Error: {e}")

real_time_collection = db["Exchange Prices"]