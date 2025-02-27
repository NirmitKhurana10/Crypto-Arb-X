from pymongo import MongoClient

try:
    client = MongoClient("mongodb://localhost:27017/ArbTradeX")
    db = client["crypto_data"]
    print("Connection established")
except Exception as e:
    print(f"Connection ERROR: {e}")

real_time_collection = db["Exchange Prices"]