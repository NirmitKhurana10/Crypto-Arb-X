from scripts.fetch_data import get_price
from backend.database.connection import db

real_time_collection = db["Exchange Prices"]

def store_real_time_data(data):
    """ Store real-time price data in MongoDB """
    real_time_collection.insert_many(data)


exchangeData = get_price()
store_real_time_data(exchangeData)

print("Real Time Data stored successfully")
