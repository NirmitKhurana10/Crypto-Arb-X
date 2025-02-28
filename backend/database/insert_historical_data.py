from scripts.fetch_historical_data import get_historical_data
from backend.database.connection import db
import pandas as pd
import ccxt
binance = ccxt.binance()


def store_historical_data(data, symbol):

    collection_name = symbol.replace("/", "_")  # Example: BTC/USDT → BTC_USDT
    historical_collection = db[collection_name]

    """ Store historical data in MongoDB """
    if isinstance(data, pd.DataFrame):  
        data_dict = data.to_dict(orient="records")  # Convert DataFrame to list of dictionaries
        historical_collection.insert_many(data_dict)
    else:
        historical_collection.insert_one(data)

TOKEN_PAIRS = ["BTC/USDT", "ETH/USDT"]

for pair in TOKEN_PAIRS:
    historical_data = get_historical_data(binance, pair)
    store_historical_data(historical_data, pair)

print("Historical Data stored in mongodb successfully")