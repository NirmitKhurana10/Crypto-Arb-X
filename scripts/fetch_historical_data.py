import pandas as pd
# Fetch historical data from Binance
def get_historical_data(exchange, symbol , timeframe="1h", limit=100):
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.strftime('%Y-%m-%d %H:%M:%S')
            df['symbol'] = symbol
            return df
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return None

# binance_data = get_historical_data(binance)
# print(binance_data.tail(10))