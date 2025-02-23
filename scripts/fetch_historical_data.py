import pandas as pd
import ccxt

binance = ccxt.binance()

# Fetch historical data from Binance
def get_historical_data(exchange, symbol="BTC/USDT", timeframe="1h", limit=100):
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.strftime('%Y-%m-%d %H:%M:%S')
    return df

binance_data = get_historical_data(binance)
print(binance_data.tail(10))