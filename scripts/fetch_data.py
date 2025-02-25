from datetime import datetime
import ccxt
import pytz

# Initialize exchange objects
binance = ccxt.binance()
coinbase = ccxt.coinbase()
kraken = ccxt.kraken()

TOKEN_PAIRS = ["BTC/USDT", "ETH/USDT"]

# Fetch current prices
def get_price():
    # Define IST timezone
    ist = pytz.timezone('Asia/Kolkata')
    # Get current time in IST
    timestamp_ist = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')

    price_data = []
    for pair in TOKEN_PAIRS:
        try:
            prices = {
                "token_pair": pair,
                "timestamp": timestamp_ist,
                "prices": {
                    "Binance": binance.fetch_ticker(pair)['last'],
                    "Coinbase": coinbase.fetch_ticker(pair)['last'],
                    "Kraken": kraken.fetch_ticker(pair)['last']
                }
            }
            price_data.append(prices)
        except Exception as e:
            print(f"Error fetching data for {pair}: {e}")
    
    return price_data