import os
from dotenv import load_dotenv
from pymongo import MongoClient

try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["crypto_data"]
    print("✅ Connected to MongoDB successfully!")
except Exception as e:
    print(f"❌ MongoDB Connection Error: {e}")

load_dotenv()
MONGO_URI = os.getenv("Mongo_URI")

client = MongoClient(MONGO_URI)
db = client["crypto_data"]

real_time_collection = db["Exchange Prices"]