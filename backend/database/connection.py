from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["crypto_data"]

real_time_collection = db["Exchange Prices"]
historical_collection = db["Historical Data"]