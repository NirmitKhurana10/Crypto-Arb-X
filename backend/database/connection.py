from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/ArbTradeX")
db = client["crypto_data"]


