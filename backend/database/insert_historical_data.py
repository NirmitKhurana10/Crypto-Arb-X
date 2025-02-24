from scripts.fetch_historical_data import get_historical_data
from backend.database.connection import historical_collection
import pandas as pd
import ccxt
binance = ccxt.binance()

TOKEN_PAIRS = ["BTC/USDT", "ETH/USDT"]



def store_historical_data(data):
    """ Store historical data in MongoDB """
    if isinstance(data, pd.DataFrame):  
        data_dict = data.to_dict(orient="records")  # Convert DataFrame to list of dictionaries
        historical_collection.insert_many(data_dict)
    else:
        historical_collection.insert_one(data)


for pair in TOKEN_PAIRS:
    historical_data = get_historical_data(binance, pair)
    store_historical_data(historical_data)

print("Historical Data stored in mongodb successfully")