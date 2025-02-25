from scripts.fetch_data import get_price
from backend.database.connection import real_time_collection


def store_real_time_data(data):
    """ Store real-time price data in MongoDB """
    real_time_collection.insert_many(data)

print("Historical Data stored successfully")

exchangeData = get_price()
store_real_time_data(exchangeData)


print("Real Time Data stored successfully")
